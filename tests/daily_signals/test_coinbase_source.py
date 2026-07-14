"""Offline tests for the Coinbase adapter: wire format, pagination, retries.

The live endpoint can't be hit from CI sandboxes, so these tests pin the
documented behavior: candles arrive as ``[time, low, high, open, close,
volume]`` rows (newest first), max 300 per request.
"""

import datetime as dt

import pandas as pd
import pytest

from daily_signals.data import coinbase
from daily_signals.data.base import DataError
from daily_signals.data.coinbase import CoinbaseSource


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise coinbase.requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Stands in for requests.Session; serves canned candles per window."""

    def __init__(self, candles_by_day, failures=0):
        self.candles_by_day = candles_by_day  # {date -> candle row}
        self.failures = failures
        self.requests = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, dict(params)))
        if self.failures > 0:
            self.failures -= 1
            return FakeResponse(status_code=503)
        start = dt.date.fromisoformat(params["start"][:10])
        end = dt.date.fromisoformat(params["end"][:10])
        rows = [
            row
            for day, row in self.candles_by_day.items()
            if start <= day <= end
        ]
        rows.sort(key=lambda r: -r[0])  # coinbase returns newest first
        return FakeResponse(payload=rows)


def candle(day: dt.date, close: float):
    ts = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())
    # [time, low, high, open, close, volume]
    return [ts, close * 0.99, close * 1.01, close - 0.5, close, 1234.5]


def make_source(candles_by_day, failures=0, sleep_off=True, monkeypatch=None):
    session = FakeSession(candles_by_day, failures=failures)
    src = CoinbaseSource(session=session)
    return src, session


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(coinbase.time, "sleep", lambda s: None)


def test_fetch_parses_wire_format_and_drops_today(monkeypatch):
    today = dt.date(2024, 3, 10)
    monkeypatch.setattr(coinbase, "utc_today", lambda: today)
    days = [today - dt.timedelta(days=i) for i in range(5)]
    src, _ = make_source({d: candle(d, 100 + i) for i, d in enumerate(days)})

    df = src.fetch("BTC-USD", start=today - dt.timedelta(days=10))

    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].is_monotonic_increasing
    # today's in-progress bar dropped; yesterday's kept
    assert df["date"].iloc[-1] == pd.Timestamp(today - dt.timedelta(days=1))
    row = df.iloc[-1]
    assert row["close"] == 101.0  # i=1 -> yesterday
    assert row["open"] == 100.5 and row["volume"] == 1234.5


def test_fetch_paginates_in_300_day_windows(monkeypatch):
    today = dt.date(2024, 12, 31)
    monkeypatch.setattr(coinbase, "utc_today", lambda: today)
    start = today - dt.timedelta(days=700)
    all_days = [start + dt.timedelta(days=i) for i in range(701)]
    src, session = make_source({d: candle(d, 50.0) for d in all_days})

    df = src.fetch("BTC-USD", start=start)

    assert len(session.requests) == 3  # 701 days / 300 per window
    for _, params in session.requests:
        w_start = dt.date.fromisoformat(params["start"][:10])
        w_end = dt.date.fromisoformat(params["end"][:10])
        assert (w_end - w_start).days <= 299
    assert len(df) == 700  # everything except today's bar
    assert df["date"].iloc[0] == pd.Timestamp(start)


def test_fetch_retries_transient_errors(monkeypatch):
    today = dt.date(2024, 3, 10)
    monkeypatch.setattr(coinbase, "utc_today", lambda: today)
    days = {today - dt.timedelta(days=1): candle(today - dt.timedelta(days=1), 77.0)}
    src, session = make_source(days, failures=2)  # two 503s, then success

    df = src.fetch("BTC-USD", start=today - dt.timedelta(days=3))
    assert df["close"].iloc[-1] == 77.0


def test_fetch_gives_up_after_retries(monkeypatch):
    today = dt.date(2024, 3, 10)
    monkeypatch.setattr(coinbase, "utc_today", lambda: today)
    src, _ = make_source({}, failures=99)
    with pytest.raises(DataError, match="failed after"):
        src.fetch("BTC-USD", start=today - dt.timedelta(days=3))


def test_unknown_product_is_a_clean_error(monkeypatch):
    today = dt.date(2024, 3, 10)
    monkeypatch.setattr(coinbase, "utc_today", lambda: today)

    class Session404(FakeSession):
        def get(self, url, params=None, timeout=None):
            return FakeResponse(status_code=404)

    src = CoinbaseSource(session=Session404({}))
    with pytest.raises(DataError, match="unknown Coinbase product"):
        src.fetch("NOPE-USD", start=today - dt.timedelta(days=3))
