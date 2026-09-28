"""Render the daily email HTML and the static web report.

Templates are jinja2; charts are (a) one matplotlib PNG for the email
(mail clients strip scripts and data: URIs, so it rides as a CID
attachment) and (b) plotly.js figures on the web page, built as plain
JSON dicts here so the light install needs no plotly package.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from daily_signals.backtest import (
    BUY_HOLD,
    ENSEMBLE,
    PORTFOLIO,
    equity_curves,
)
from daily_signals.state import StateStore

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Chart palette — the validated reference instance (see the dataviz method):
# series-1 blue carries the strategy, neutral ink carries the benchmark.
COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series1": "#2a78d6",
    "good": "#0ca30c",
    "critical": "#d03b3b",
}

VOICE_LABELS = {
    "trend_sma": "Trend (SMA cross)",
    "momentum_roc": "Momentum (ROC)",
    "meanrev_rsi": "Mean reversion (RSI)",
    "rl_ppo": "RL agent (PPO)",
    ENSEMBLE: "Ensemble",
}

DISCLAIMER = (
    "This is research output from an automated system, provided for "
    "informational purposes only. It is not investment advice, not an offer "
    "or solicitation, and past (simulated) performance does not guarantee "
    "future results. Trading cryptocurrencies involves substantial risk of "
    "loss. Do your own research."
)


@dataclass
class ReportArtifacts:
    as_of: str
    site_index: Path
    email_subject: str
    email_html: str
    email_text: str
    chart_png: Optional[bytes]


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float, digits: int = 1, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    sign = "+" if signed else ""
    return f"{value * 100:{sign}.{digits}f}%"


def _stance_label(stance: int) -> str:
    return "LONG" if int(stance) == 1 else "FLAT"


def _build_stance_rows(cfg, results, data, as_of) -> List[dict]:
    rows = []
    day = pd.Timestamp(as_of)
    for asset in cfg.assets:
        result = results[asset.symbol]
        df = data[asset.symbol]
        close = float(df.loc[df["date"] <= day, "close"].iloc[-1])
        rows.append({
            "symbol": asset.symbol,
            "stance": _stance_label(result.stance),
            "is_long": result.stance == 1,
            "changed": result.changed,
            "prev": _stance_label(result.prev_stance),
            "close": _fmt_money(close),
            "weight": f"{asset.weight:.0%}",
            "vote": result.vote_summary(),
        })
    return rows


def _build_voice_rows(voices_by_asset) -> List[dict]:
    rows = []
    for symbol, voices in voices_by_asset.items():
        for voice in voices:
            rows.append({
                "symbol": symbol,
                "voice": VOICE_LABELS.get(voice.name, voice.name),
                "stance": _stance_label(voice.stance) if voice.available else "—",
                "is_long": voice.available and voice.stance == 1,
                "available": voice.available,
                "weight": f"{voice.weight:g}",
                "rationale": voice.rationale,
            })
    return rows


def _build_paper_summary(cfg, store: StateStore, data, as_of) -> Optional[dict]:
    equity = store.read_equity()
    if equity.empty:
        return None
    initial = cfg.portfolio.initial_cash
    latest = equity.iloc[-1]
    nav = float(latest["nav"])
    bh_nav = float(latest["buy_hold_nav"])
    cash, positions = store.holdings()
    day = pd.Timestamp(as_of)
    position_rows = []
    for symbol, qty in sorted(positions.items()):
        df = data.get(symbol)
        price = float(df.loc[df["date"] <= day, "close"].iloc[-1]) if df is not None else 0.0
        position_rows.append({
            "symbol": symbol,
            "qty": f"{qty:,.6f}",
            "value": _fmt_money(qty * price),
        })
    ledger = store.read_ledger()
    trades = ledger[ledger["side"].isin(["BUY", "SELL"])]
    return {
        "nav": _fmt_money(nav),
        "initial": _fmt_money(initial),
        "total_return": _fmt_pct(nav / initial - 1),
        "total_return_positive": nav >= initial,
        "bh_nav": _fmt_money(bh_nav),
        "vs_bh": _fmt_pct((nav - bh_nav) / initial),
        "vs_bh_positive": nav >= bh_nav,
        "days": len(equity),
        "since": equity["date"].iloc[0].strftime("%Y-%m-%d"),
        "cash": _fmt_money(cash),
        "positions": position_rows,
        "num_trades": len(trades),
    }


def _build_recent_trades(store: StateStore, limit: int = 10) -> List[dict]:
    ledger = store.read_ledger()
    trades = ledger[ledger["side"].isin(["BUY", "SELL"])].tail(limit)
    rows = []
    for _, row in trades.iloc[::-1].iterrows():
        rows.append({
            "date": row["date"],
            "asset": row["asset"],
            "side": row["side"],
            "qty": f"{float(row['qty']):,.6f}",
            "price": _fmt_money(float(row["fill_price"])),
            "fee": _fmt_money(float(row["fee"])),
        })
    return rows


def _build_history(store: StateStore, cfg, days: int = 14) -> List[dict]:
    symbols = [a.symbol for a in cfg.assets]
    by_date: Dict[str, Dict[str, str]] = {}
    for row in store.read_signals():
        if row["voice"] != ENSEMBLE:
            continue
        entry = by_date.setdefault(row["date"], {})
        entry[row["asset"]] = _stance_label(row["stance"])
    history = []
    for day in sorted(by_date, reverse=True)[:days]:
        history.append({
            "date": day,
            "stances": [by_date[day].get(s, "—") for s in symbols],
        })
    return history


def _format_tearsheet(sheet: pd.DataFrame) -> List[dict]:
    rows = []
    for _, row in sheet.iterrows():
        rows.append({
            "asset": row["asset"],
            "voice": VOICE_LABELS.get(row["voice"], row["voice"]),
            "is_ensemble": row["voice"] == ENSEMBLE,
            "is_benchmark": row["voice"] == BUY_HOLD,
            "total_return": _fmt_pct(row["total_return"]),
            "cagr": _fmt_pct(row["cagr"]),
            "sharpe": "—" if pd.isna(row["sharpe"]) else f"{row['sharpe']:.2f}",
            "max_drawdown": _fmt_pct(row["max_drawdown"]),
            "win_rate": _fmt_pct(row["win_rate"], signed=False),
            "trades_per_month": "—" if pd.isna(row["trades_per_month"])
            else f"{row['trades_per_month']:.2f}",
            "vs_buy_hold": _fmt_pct(row.get("vs_buy_hold")),
        })
    return rows


def _headline(sheet: pd.DataFrame) -> Optional[dict]:
    match = sheet[(sheet["asset"] == PORTFOLIO) & (sheet["voice"] == ENSEMBLE)]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "cagr": _fmt_pct(row["cagr"]),
        "sharpe": f"{row['sharpe']:.2f}",
        "max_drawdown": _fmt_pct(row["max_drawdown"]),
        "vs_buy_hold": _fmt_pct(row["vs_buy_hold"]),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _line_figure(series_map: Dict[str, pd.Series], title: str,
                 y_title: str) -> dict:
    """A plotly figure dict (rendered by plotly.js on the web page)."""
    data = []
    palette = [COLORS["series1"], COLORS["muted"], "#1baf7a", "#eda100"]
    for i, (name, series) in enumerate(series_map.items()):
        data.append({
            "type": "scatter",
            "mode": "lines",
            "name": name,
            "x": [d.strftime("%Y-%m-%d") for d in series.index],
            "y": [round(float(v), 6) for v in series.values],
            "line": {
                "color": palette[i % len(palette)],
                "width": 2,
                "dash": "dot" if "buy" in name.lower() else "solid",
            },
        })
    layout = {
        "title": {"text": title, "font": {"size": 15, "color": COLORS["ink"]}},
        "paper_bgcolor": COLORS["surface"],
        "plot_bgcolor": COLORS["surface"],
        "font": {"color": COLORS["ink_secondary"], "size": 12},
        "margin": {"l": 60, "r": 20, "t": 48, "b": 40},
        "xaxis": {"gridcolor": COLORS["grid"], "linecolor": COLORS["axis"]},
        "yaxis": {"gridcolor": COLORS["grid"], "linecolor": COLORS["axis"],
                  "title": {"text": y_title}},
        "legend": {"orientation": "h", "y": -0.18},
        "hovermode": "x unified",
    }
    return {"data": data, "layout": layout}


def build_site_charts(cfg, store: StateStore, data, sheet) -> Dict[str, str]:
    charts: Dict[str, str] = {}
    equity = store.read_equity()
    if len(equity) >= 2:
        idx = equity["date"]
        charts["paper"] = json.dumps(_line_figure(
            {
                "Paper portfolio": pd.Series(equity["nav"].values, index=idx),
                "Buy & hold": pd.Series(equity["buy_hold_nav"].values, index=idx),
            },
            "Paper portfolio vs buy & hold (live track record)",
            "NAV ($)",
        ))
    curves = equity_curves(cfg, data)
    portfolio_cols = [c for c in curves.columns if c.startswith(PORTFOLIO)]
    if portfolio_cols:
        charts["backtest"] = json.dumps(_line_figure(
            {
                "Ensemble (backtest)": curves[f"{PORTFOLIO} ensemble"],
                "Buy & hold": curves[f"{PORTFOLIO} buy&hold"],
            },
            f"Blended portfolio backtest since {cfg.backtest.start} (growth of $1)",
            "Growth of $1",
        ))
    return charts


def equity_chart_png(cfg, store: StateStore, data, lookback_days: int = 180
                     ) -> Optional[bytes]:
    """The email chart: live track record once it exists, else the backtest."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    equity = store.read_equity()
    if len(equity) >= 2:
        x = equity["date"]
        strategy = pd.Series(equity["nav"].values, index=x)
        bench = pd.Series(equity["buy_hold_nav"].values, index=x)
        title = "Paper portfolio vs buy & hold"
        y_label = "NAV ($)"
    else:
        curves = equity_curves(cfg, data)
        if f"{PORTFOLIO} ensemble" not in curves.columns or len(curves) < 2:
            return None
        tail = curves.tail(lookback_days)
        strategy = tail[f"{PORTFOLIO} ensemble"]
        bench = tail[f"{PORTFOLIO} buy&hold"]
        title = f"Backtest, last {len(tail)} days (growth of $1)"
        y_label = "Growth of $1"

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    ax.plot(strategy.index, strategy.values, color=COLORS["series1"],
            linewidth=2, label="Strategy")
    ax.plot(bench.index, bench.values, color=COLORS["muted"], linewidth=2,
            linestyle=(0, (3, 2)), label="Buy & hold")
    ax.set_title(title, color=COLORS["ink"], fontsize=12, loc="left")
    ax.set_ylabel(y_label, color=COLORS["ink_secondary"], fontsize=9)
    ax.grid(color=COLORS["grid"], linewidth=0.75)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLORS["ink_secondary"])
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_reports(cfg, as_of, results, voices_by_asset, store: StateStore,
                   data, sheet, trades, stale: bool = False) -> ReportArtifacts:
    env = _environment()
    as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    from daily_signals.state import git_sha

    stance_rows = _build_stance_rows(cfg, results, data, as_of)
    context = {
        "title": cfg.report.title,
        "as_of": as_of_str,
        "generated": generated,
        "git_sha": git_sha(),
        "stale": stale,
        "stances": stance_rows,
        "voices": _build_voice_rows(voices_by_asset),
        "paper": _build_paper_summary(cfg, store, data, as_of),
        "recent_trades": _build_recent_trades(store),
        "history": _build_history(store, cfg),
        "asset_symbols": [a.symbol for a in cfg.assets],
        "tearsheet": _format_tearsheet(sheet),
        "headline": _headline(sheet),
        "backtest_start": cfg.backtest.start,
        "commission_bps": cfg.backtest.commission_bps,
        "site_url": cfg.report.site_url,
        "disclaimer": DISCLAIMER,
        "colors": COLORS,
        "trades_today": trades,
    }

    chart_png = equity_chart_png(cfg, store, data)
    context["has_chart"] = chart_png is not None

    email_html = env.get_template("email.html.j2").render(**context)
    email_text = _plain_text_summary(context)

    site_dir = Path(cfg.report.site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "archive").mkdir(exist_ok=True)

    site_charts = build_site_charts(cfg, store, data, sheet)
    archive_names = sorted(
        p.stem for p in (site_dir / "archive").glob("*.html")
    )
    template = env.get_template("site.html.j2")
    index_html = template.render(
        **context, charts=site_charts, root_prefix="",
        archive_names=list(reversed(archive_names[-30:])),
    )
    site_index = site_dir / "index.html"
    site_index.write_text(index_html)

    archive_html = template.render(
        **context, charts=site_charts, root_prefix="../", archive_names=[],
        is_archive=True,
    )
    (site_dir / "archive" / f"{as_of_str}.html").write_text(archive_html)

    changes = [r["symbol"] for r in stance_rows if r["changed"]]
    subject = f"{as_of_str} — " + ", ".join(
        f"{r['symbol'].split('-')[0]} {r['stance']}" for r in stance_rows
    )
    if stale:
        subject += " (STALE DATA)"
    elif changes:
        subject += f" — changed: {', '.join(changes)}"

    return ReportArtifacts(
        as_of=as_of_str,
        site_index=site_index,
        email_subject=subject,
        email_html=email_html,
        email_text=email_text,
        chart_png=chart_png,
    )


def _plain_text_summary(ctx: dict) -> str:
    lines = [f"{ctx['title']} — {ctx['as_of']}", ""]
    if ctx["stale"]:
        lines.append("*** STALE DATA — no new signal today ***")
        lines.append("")
    for row in ctx["stances"]:
        marker = " (changed)" if row["changed"] else ""
        lines.append(f"{row['symbol']}: {row['stance']}{marker} — {row['vote']}")
    paper = ctx.get("paper")
    if paper:
        lines += [
            "",
            f"Paper portfolio: {paper['nav']} ({paper['total_return']} since "
            f"{paper['since']}; buy & hold {paper['bh_nav']})",
        ]
    if ctx.get("site_url"):
        lines += ["", f"Full report: {ctx['site_url']}"]
    lines += ["", ctx["disclaimer"]]
    return "\n".join(lines)
