"""Offline tests for the data layer: normalization, validation, cache merge."""

import datetime as dt

import pandas as pd
import pytest

from daily_signals.data import cache
from daily_signals.data.base import (
    DataError,
    gap_report,
    normalize_frame,
    validate_frame,
)


def make_frame(start="2024-01-01", days=10, close_start=100.0):
    dates = pd.date_range(start, periods=days, freq="D")
    closes = [close_start + i for i in range(days)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000.0] * days,
        }
    )


class StubSource:
    """OHLCVSource double returning a canned frame, recording the start arg."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def fetch(self, symbol, start=None):
        self.calls.append(start)
        if start is None:
            return self.frame
        return self.frame[self.frame["date"] >= pd.Timestamp(start)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# normalize / validate
# ---------------------------------------------------------------------------

def test_normalize_drops_in_progress_bar():
    df = make_frame("2024-01-01", days=5)
    today = dt.date(2024, 1, 5)  # last row IS today -> in progress -> dropped
    out = normalize_frame(df, "BTC-USD", today=today)
    assert len(out) == 4
    assert out["date"].iloc[-1] == pd.Timestamp("2024-01-04")


def test_normalize_sorts_and_dedupes_keeping_last():
    df = make_frame("2024-01-01", days=4)
    dup = df.iloc[[2]].assign(close=999.0)
    scrambled = pd.concat([df.iloc[::-1], dup], ignore_index=True)
    out = normalize_frame(scrambled, "X", today=dt.date(2024, 2, 1))
    assert out["date"].is_monotonic_increasing
    assert len(out) == 4
    assert out.loc[out["date"] == "2024-01-03", "close"].item() == 999.0


def test_validate_rejects_duplicates_and_disorder():
    good = make_frame()
    validate_frame(good, "X")

    dup = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataError, match="duplicate"):
        validate_frame(dup, "X")

    disordered = good.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataError, match="ascending"):
        validate_frame(disordered, "X")

    nan_close = good.copy()
    nan_close.loc[3, "close"] = float("nan")
    with pytest.raises(DataError, match="NaN"):
        validate_frame(nan_close, "X")


def test_gap_report_finds_missing_days():
    df = make_frame(days=6)
    df = df[df["date"] != pd.Timestamp("2024-01-03")]
    assert gap_report(df) == [dt.date(2024, 1, 3)]
    assert gap_report(make_frame()) == []


# ---------------------------------------------------------------------------
# cache round-trip and incremental refresh
# ---------------------------------------------------------------------------

def make_config(tmp_path, source="coinbase"):
    from daily_signals.config import (
        AssetConfig, BacktestConfig, Config, DataConfig, EmailConfig,
        EnsembleConfig, PortfolioConfig, ReportConfig, RLConfig, StrategySpec,
    )

    return Config(
        base_currency="USD",
        assets=(AssetConfig(symbol="BTC-USD", weight=1.0),),
        data=DataConfig(source=source, cache_dir=str(tmp_path / "cache")),
        strategies=(StrategySpec(name="trend_sma"),),
        rl=RLConfig(),
        ensemble=EnsembleConfig(),
        backtest=BacktestConfig(start="2024-01-01"),
        portfolio=PortfolioConfig(),
        report=ReportConfig(),
        email=EmailConfig(),
        state_dir=str(tmp_path / "state"),
    )


def test_write_read_round_trip(tmp_path):
    df = make_frame()
    path = tmp_path / "BTC-USD_1d.csv"
    cache.write_frame(path, df)
    out = cache.read_frame(path, "BTC-USD")
    pd.testing.assert_frame_equal(out, df)


def test_refresh_bootstrap_then_incremental(tmp_path, no_network):
    cfg = make_config(tmp_path)
    full = make_frame("2024-01-01", days=20)
    source = StubSource(full)

    first = cache.refresh(cfg, "BTC-USD", source=source)
    assert len(first) == 20
    # bootstrap fetch starts from the configured backtest start
    assert source.calls[0] == dt.date(2024, 1, 1)

    # upstream corrects the last cached bar and adds two new ones
    revised = make_frame("2024-01-01", days=22)
    revised.loc[19, "close"] = 555.0
    source = StubSource(revised)
    second = cache.refresh(cfg, "BTC-USD", source=source)

    # incremental start = last cached date minus the overlap window
    assert source.calls[0] == dt.date(2024, 1, 20) - dt.timedelta(
        days=cache.REFRESH_OVERLAP_DAYS
    )
    assert len(second) == 22
    assert second.loc[second["date"] == "2024-01-20", "close"].item() == 555.0

    on_disk = cache.read_frame(cache.cache_path(cfg, "BTC-USD"), "BTC-USD")
    assert len(on_disk) == 22


def test_load_without_cache_raises(tmp_path, no_network):
    cfg = make_config(tmp_path)
    with pytest.raises(DataError, match="daily-signals fetch"):
        cache.load(cfg, "BTC-USD")


def test_fixture_mode_loads_committed_fixtures(fixture_config, no_network):
    data = cache.load_all(fixture_config)
    assert set(data) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    for symbol, df in data.items():
        validate_frame(df, symbol)
        assert len(df) == 500
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    # all fixtures share the same date range so the pipeline as-of aligns
    last_dates = {df["date"].iloc[-1] for df in data.values()}
    assert last_dates == {pd.Timestamp("2019-10-17")}
