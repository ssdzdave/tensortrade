"""End-to-end: `run-daily --dry-run` on fixtures, fully offline.

Asserts the pipeline appends the right state rows, renders the site and the
outbox .eml, and that a second invocation is a pure no-op on state.
"""

import json
from pathlib import Path

import pytest

from daily_signals import cli


def run_cli(config_path, *argv):
    return cli.main(["--config", str(config_path), *argv])


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


@pytest.fixture
def state_paths(fixture_config, tmp_path):
    return {
        "config": tmp_path / "signals.yaml",
        "state": Path(fixture_config.state_dir),
        "site": Path(fixture_config.report.site_dir),
        "outbox": Path(fixture_config.report.outbox_dir),
    }


def test_run_daily_end_to_end_and_idempotent(state_paths, no_network, capsys):
    config = state_paths["config"]

    assert run_cli(config, "run-daily", "--dry-run") == 0

    # --- signals: one row per voice per asset + one ensemble row per asset ---
    signals = read_lines(state_paths["state"] / "signals.jsonl")
    assert len(signals) == 3 * 4  # 3 assets x (3 strategy voices + ensemble)
    as_of = signals[0]["date"]
    assert as_of == "2019-10-17"  # last fixture bar
    ensemble_rows = [s for s in signals if s["voice"] == "ensemble"]
    assert {s["asset"] for s in ensemble_rows} == {"BTC-USD", "ETH-USD", "SOL-USD"}
    for row in signals:
        assert row["rationale"]
        assert row["generated_at_utc"]

    # --- ledger: deposit plus a buy for each LONG stance ---
    ledger = (state_paths["state"] / "ledger.csv").read_text().splitlines()
    header, rows = ledger[0], ledger[1:]
    assert header.startswith("date,asset,side")
    sides = [r.split(",")[2] for r in rows]
    assert sides[0] == "DEPOSIT"
    long_assets = {s["asset"] for s in ensemble_rows if s["stance"] == 1}
    assert sides.count("BUY") == len(long_assets)

    # --- equity: exactly one mark ---
    equity = (state_paths["state"] / "equity.csv").read_text().splitlines()
    assert len(equity) == 2  # header + one row
    assert equity[1].startswith(as_of)

    # --- reports ---
    index_html = (state_paths["site"] / "index.html").read_text()
    assert as_of in index_html
    assert "Not investment advice" not in index_html  # full disclaimer text used
    assert "informational purposes only" in index_html
    assert "Today's stances" in index_html or "Today&#39;s stances" in index_html
    for symbol in ("BTC-USD", "ETH-USD", "SOL-USD"):
        assert symbol in index_html
    assert (state_paths["site"] / "archive" / f"{as_of}.html").exists()

    # --- outbox email ---
    eml = state_paths["outbox"] / f"{as_of}.eml"
    assert eml.exists()
    raw = eml.read_bytes()
    assert b"Subject:" in raw and b"text/html" in raw

    # --- second run: state unchanged, still exit 0 ---
    before = {
        p.name: p.read_bytes()
        for p in state_paths["state"].iterdir()
    }
    assert run_cli(config, "run-daily", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "already recorded" in out
    after = {p.name: p.read_bytes() for p in state_paths["state"].iterdir()}
    assert before == after


def test_backtest_subcommand_prints_tearsheet(state_paths, no_network, capsys):
    assert run_cli(state_paths["config"], "backtest") == 0
    out = capsys.readouterr().out
    assert "PORTFOLIO" in out
    assert "trend_sma" in out


def test_render_subcommand_rewrites_site_without_state_changes(
        state_paths, no_network, capsys):
    config = state_paths["config"]
    assert run_cli(config, "run-daily", "--dry-run") == 0
    signals_before = (state_paths["state"] / "signals.jsonl").read_bytes()

    (state_paths["site"] / "index.html").unlink()
    assert run_cli(config, "render") == 0
    assert (state_paths["site"] / "index.html").exists()
    assert (state_paths["state"] / "signals.jsonl").read_bytes() == signals_before


def test_missing_config_is_clean_error(tmp_path, capsys):
    code = cli.main(["--config", str(tmp_path / "nope.yaml"), "fetch"])
    assert code == 2
    err = capsys.readouterr().err
    assert "config error" in err
    assert "daily-signals init" in err
