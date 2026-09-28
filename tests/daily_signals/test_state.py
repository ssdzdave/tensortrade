"""State store: append-only files, idempotency, ledger-derived holdings."""

import datetime as dt

import pytest

from daily_signals.ensemble import Voice, combine_voices
from daily_signals.state import StateStore


def make_voices(stance=1):
    return [
        Voice("trend_sma", stance, 1.0, "trend rationale"),
        Voice("momentum_roc", stance, 1.0, "momentum rationale"),
    ]


def make_results(voices, prev=0):
    return combine_voices(voices, prev_stance=prev)


def test_append_signals_and_idempotency(tmp_path):
    store = StateStore(tmp_path / "state")
    voices = {"BTC-USD": make_voices(1)}
    results = {"BTC-USD": make_results(voices["BTC-USD"])}
    day = dt.date(2024, 1, 2)

    added = store.append_signals(day, voices, results)
    assert added == 3  # two voices + ensemble

    rows = store.read_signals()
    assert len(rows) == 3
    ensemble = [r for r in rows if r["voice"] == "ensemble"][0]
    assert ensemble["stance"] == 1
    assert ensemble["date"] == "2024-01-02"
    assert "generated_at_utc" in ensemble

    # same day again: nothing appended
    assert store.append_signals(day, voices, results) == 0
    assert len(store.read_signals()) == 3

    # next day appends fresh rows
    assert store.append_signals(dt.date(2024, 1, 3), voices, results) == 3
    assert store.has_ensemble_rows(dt.date(2024, 1, 3))
    assert not store.has_ensemble_rows(dt.date(2024, 1, 4))


def test_latest_stances_takes_most_recent(tmp_path):
    store = StateStore(tmp_path / "state")
    long_voices = {"BTC-USD": make_voices(1)}
    flat_voices = {"BTC-USD": make_voices(0)}
    store.append_signals(dt.date(2024, 1, 2), long_voices,
                         {"BTC-USD": make_results(long_voices["BTC-USD"])})
    store.append_signals(dt.date(2024, 1, 3), flat_voices,
                         {"BTC-USD": make_results(flat_voices["BTC-USD"], prev=1)})
    assert store.latest_stances() == {"BTC-USD": 0}


def test_holdings_from_ledger_math(tmp_path):
    store = StateStore(tmp_path / "state")
    store.append_ledger_rows([
        {"date": "2024-01-02", "asset": "CASH", "side": "DEPOSIT", "qty": 10_000,
         "fill_price": 1.0, "fee": 0.0, "cash_after": 10_000, "note": "seed"},
        {"date": "2024-01-02", "asset": "BTC-USD", "side": "BUY", "qty": 0.1,
         "fill_price": 50_000, "fee": 7.5, "cash_after": 4_992.5, "note": ""},
    ])
    cash, positions = store.holdings()
    assert cash == pytest.approx(10_000 - 0.1 * 50_000 - 7.5)
    assert positions == {"BTC-USD": pytest.approx(0.1)}

    store.append_ledger_rows([
        {"date": "2024-01-05", "asset": "BTC-USD", "side": "SELL", "qty": 0.1,
         "fill_price": 60_000, "fee": 9.0, "cash_after": 10_983.5, "note": ""},
    ])
    cash, positions = store.holdings()
    assert cash == pytest.approx(4_992.5 + 0.1 * 60_000 - 9.0)
    assert positions == {}


def test_ledger_append_skips_same_day_same_asset(tmp_path):
    store = StateStore(tmp_path / "state")
    row = {"date": "2024-01-02", "asset": "BTC-USD", "side": "BUY", "qty": 1,
           "fill_price": 100, "fee": 0.1, "cash_after": 900, "note": ""}
    assert store.append_ledger_rows([row]) == 1
    assert store.append_ledger_rows([row]) == 0
    assert len(store.read_ledger()) == 1


def test_equity_append_guarded_by_date(tmp_path):
    store = StateStore(tmp_path / "state")
    row = {"date": "2024-01-02", "nav": 10_000, "cash": 10_000,
           "positions_value": 0.0, "buy_hold_nav": 10_000}
    assert store.append_equity_row(row)
    assert not store.append_equity_row({**row, "nav": 99})
    equity = store.read_equity()
    assert len(equity) == 1
    assert equity["nav"].iloc[0] == 10_000
    assert store.inception().strftime("%Y-%m-%d") == "2024-01-02"
