"""Report rendering and email construction, offline."""

import email
from email import policy

import pytest

from daily_signals.backtest import tearsheet
from daily_signals.data.cache import load_all
from daily_signals.emailer import build_message, send_email
from daily_signals.ensemble import combine_voices, compute_voices
from daily_signals.portfolio import update_paper_portfolio
from daily_signals.report.render import render_reports
from daily_signals.state import StateStore


@pytest.fixture
def rendered(fixture_config, no_network):
    cfg = fixture_config
    data = load_all(cfg)
    as_of = min(df["date"].iloc[-1] for df in data.values()).date()
    store = StateStore(cfg.state_dir)
    voices_by_asset = compute_voices(cfg, data)
    results = {
        a.symbol: combine_voices(voices_by_asset[a.symbol], prev_stance=0,
                                 long_threshold=cfg.ensemble.long_threshold)
        for a in cfg.assets
    }
    store.append_signals(as_of, voices_by_asset, results)
    trades = update_paper_portfolio(cfg, store, as_of, results, data)
    sheet = tearsheet(cfg, data)
    artifacts = render_reports(cfg, as_of=as_of, results=results,
                               voices_by_asset=voices_by_asset, store=store,
                               data=data, sheet=sheet, trades=trades)
    return cfg, artifacts


def test_email_html_contains_stances_voices_disclaimer(rendered):
    _, artifacts = rendered
    html = artifacts.email_html
    for must_have in ("BTC-USD", "ETH-USD", "SOL-USD", "Disclaimer",
                      "informational purposes only", "2019-10-17",
                      "Trend (SMA cross)", "Momentum (ROC)"):
        assert must_have in html, must_have
    assert artifacts.email_subject.startswith("2019-10-17")
    assert "BTC LONG" in artifacts.email_subject or "BTC FLAT" in artifacts.email_subject


def test_email_text_fallback_is_readable(rendered):
    _, artifacts = rendered
    text = artifacts.email_text
    assert "BTC-USD" in text
    assert "informational purposes only" in text


def test_chart_png_produced_from_backtest_on_day_one(rendered):
    _, artifacts = rendered
    # only one equity mark exists, so the chart falls back to the backtest
    assert artifacts.chart_png is not None
    assert artifacts.chart_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_site_contains_charts_tearsheet_and_archive(rendered):
    cfg, artifacts = rendered
    html = artifacts.site_index.read_text()
    assert "chart-backtest" in html  # plotly div present
    assert "cdn.plot.ly" in html
    assert "PORTFOLIO" in html
    assert "Signal history" in html
    archive = artifacts.site_index.parent / "archive" / f"{artifacts.as_of}.html"
    assert archive.exists()
    assert "latest report" in archive.read_text()


def test_build_message_structure_with_cid_chart(rendered):
    cfg, artifacts = rendered
    msg = build_message(cfg, artifacts)
    assert msg["Subject"].startswith(cfg.email.subject_prefix)
    assert msg["To"] == cfg.email.from_addr
    assert "Bcc" not in msg  # recipients never leak into headers

    parsed = email.message_from_bytes(bytes(msg), policy=policy.default)
    parts = {p.get_content_type() for p in parsed.walk()}
    assert "text/plain" in parts
    assert "text/html" in parts
    assert "image/png" in parts  # CID chart attached to the html part


def test_send_email_dry_run_writes_eml(rendered, tmp_path):
    cfg, artifacts = rendered
    path = send_email(cfg, artifacts, dry_run=True)
    assert path is not None and path.exists()
    assert path.suffix == ".eml"


def test_send_email_without_recipients_raises(rendered, monkeypatch):
    cfg, artifacts = rendered
    with pytest.raises(ValueError, match="bcc is empty"):
        send_email(cfg, artifacts, dry_run=False)
