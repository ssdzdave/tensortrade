"""Command-line interface and daily-run orchestrator.

Subcommands import their dependencies lazily so that ``--help`` and light
commands stay fast and the package never drags in heavy libraries.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_CONFIG = "config/signals.yaml"
EXAMPLE_CONFIG = "config/signals.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daily-signals",
        description="Daily crypto trading signals: fetch data, compute the "
        "strategy ensemble, track a paper portfolio and deliver reports.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help=f"Path to YAML config (default: {DEFAULT_CONFIG})"
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create config from the example, fetch history, run a backtest")
    p_init.set_defaults(func=cmd_init)

    p_fetch = sub.add_parser("fetch", help="Refresh the local OHLCV cache for all configured assets")
    p_fetch.set_defaults(func=cmd_fetch)

    p_bt = sub.add_parser("backtest", help="Run the walk-forward backtest and print the tearsheet")
    p_bt.set_defaults(func=cmd_backtest)

    p_run = sub.add_parser("run-daily", help="Full daily pipeline: fetch, signals, ledger, reports, email")
    p_run.add_argument("--dry-run", action="store_true", help="Do not send email; write .eml to the outbox instead")
    p_run.add_argument("--resend", action="store_true", help="Re-send email even if today's signals already exist")
    p_run.set_defaults(func=cmd_run_daily)

    p_render = sub.add_parser("render", help="Re-render the web report and email HTML from current state")
    p_render.set_defaults(func=cmd_render)

    p_send = sub.add_parser("send-email", help="Render and send (or outbox) the latest daily email")
    p_send.add_argument("--dry-run", action="store_true", help="Write .eml to the outbox instead of sending")
    p_send.set_defaults(func=cmd_send_email)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args) or 0
    except Exception as exc:  # surface a clean one-line error for operators
        from daily_signals.config import ConfigError

        if isinstance(exc, ConfigError):
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        raise


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_init(args) -> int:
    from daily_signals.config import load_config

    config_path = Path(args.config)
    if not config_path.exists():
        example = Path(EXAMPLE_CONFIG)
        if not example.exists():
            print(f"error: {example} not found; are you running from the repo root?", file=sys.stderr)
            return 1
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(example, config_path)
        print(f"Created {config_path} from {example} — edit it, then re-run.")
    cfg = load_config(config_path)

    from daily_signals.data.cache import refresh_all

    print("Fetching history for:", ", ".join(a.symbol for a in cfg.assets))
    data = refresh_all(cfg)
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} daily bars ({df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()})")

    return cmd_backtest(args)


def cmd_fetch(args) -> int:
    from daily_signals.config import load_config
    from daily_signals.data.cache import refresh_all

    cfg = load_config(args.config)
    data = refresh_all(cfg)
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} bars, last close {df['close'].iloc[-1]:.2f} on {df['date'].iloc[-1].date()}")
    return 0


def cmd_backtest(args) -> int:
    from daily_signals.backtest import render_tearsheet_text, tearsheet
    from daily_signals.config import load_config
    from daily_signals.data.cache import load_all

    cfg = load_config(args.config)
    data = load_all(cfg)
    sheet = tearsheet(cfg, data)
    print(render_tearsheet_text(sheet))
    return 0


def cmd_run_daily(args) -> int:
    from daily_signals.config import load_config

    cfg = load_config(args.config)
    return run_daily(cfg, dry_run=args.dry_run, resend=args.resend)


def cmd_render(args) -> int:
    from daily_signals.config import load_config

    cfg = load_config(args.config)
    return run_daily(cfg, dry_run=True, render_only=True)


def cmd_send_email(args) -> int:
    from daily_signals.config import load_config

    cfg = load_config(args.config)
    return run_daily(cfg, dry_run=args.dry_run, resend=True, refresh=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_daily(cfg, dry_run: bool = False, resend: bool = False,
              render_only: bool = False, refresh: bool = True) -> int:
    """The daily pipeline. Returns a process exit code.

    Steps: refresh cache -> compute voices -> ensemble -> book paper trades ->
    append state -> render site + email -> send. Appends are idempotent: the
    run is keyed by the as-of date (last closed UTC bar); re-running on the
    same day re-renders but appends nothing and, unless ``resend``, does not
    re-email.
    """
    from daily_signals.backtest import tearsheet
    from daily_signals.data.cache import load_all, refresh_all
    from daily_signals.ensemble import (
        combine_voices,
        compute_voices,
        rl_stance_series,
    )
    from daily_signals.portfolio import update_paper_portfolio
    from daily_signals.report.render import render_reports
    from daily_signals.state import StateStore

    stale = False
    if refresh and not render_only and cfg.data.source != "fixture":
        try:
            data = refresh_all(cfg)
        except Exception as exc:
            print(f"warning: data refresh failed ({exc}); falling back to cache", file=sys.stderr)
            data = load_all(cfg)
            stale = True
    else:
        data = load_all(cfg)

    as_of = min(df["date"].iloc[-1] for df in data.values()).date()
    store = StateStore(cfg.state_dir)
    already_ran = store.has_ensemble_rows(as_of)

    # 1. Voices per asset (rule strategies + optional RL), then the ensemble.
    voices_by_asset = compute_voices(cfg, data)
    prev_stances = store.latest_stances()
    results = {}
    for asset in cfg.assets:
        results[asset.symbol] = combine_voices(
            voices_by_asset[asset.symbol],
            prev_stance=prev_stances.get(asset.symbol, 0),
            long_threshold=cfg.ensemble.long_threshold,
        )

    # 2. Persist signals + paper portfolio state (skipped when idempotent/stale).
    trades = []
    if not render_only and not stale and not already_ran:
        store.append_signals(as_of, voices_by_asset, results)
        trades = update_paper_portfolio(cfg, store, as_of, results, data)
    elif already_ran and not render_only:
        print(f"signals for {as_of} already recorded; re-rendering only")

    # 3. Reports (always re-rendered so the site reflects the latest state).
    rl_series = rl_stance_series(cfg, data)
    extra = ({s: {"rl_ppo": ser} for s, ser in rl_series.items()}
             if rl_series else None)
    sheet = tearsheet(cfg, data, extra_series=extra)
    artifacts = render_reports(
        cfg,
        as_of=as_of,
        results=results,
        voices_by_asset=voices_by_asset,
        store=store,
        data=data,
        sheet=sheet,
        trades=trades,
        stale=stale,
    )
    print(f"site: {artifacts.site_index}")

    # 4. Email.
    if render_only:
        return 0
    should_email = cfg.email.enabled and (not already_ran or resend)
    if not should_email and not dry_run:
        if not cfg.email.enabled:
            print("email disabled (email.enabled: false); "
                  "use --dry-run to write an .eml to the outbox")
        return 0

    from daily_signals.emailer import send_email

    outbox = send_email(cfg, artifacts, dry_run=dry_run or not cfg.email.enabled)
    if outbox:
        print(f"email written to outbox: {outbox}")
    else:
        print(f"email sent to {len(cfg.email.bcc)} recipient(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
