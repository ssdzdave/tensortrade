"""Strategy tests on synthetic series with hand-known signal transitions."""

import numpy as np
import pandas as pd
import pytest

from daily_signals.strategies import build_strategies
from daily_signals.strategies.meanrev import RsiMeanReversion, rsi
from daily_signals.strategies.momentum import RocMomentum
from daily_signals.strategies.trend import SmaCross


def frame(closes):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1000.0,
        }
    )


class TestSmaCross:
    def test_hand_computed_crossings(self):
        # fast=2, slow=4 on a V-shape: down then up
        df = frame([10, 9, 8, 7, 6, 7, 8, 9, 10, 11])
        strat = SmaCross(fast=2, slow=4)
        out = strat.stance_series(df).tolist()
        # warmup (first 3 bars lack SMA4) -> 0; downtrend -> 0
        # closes[5..]: fast sma [6.5, 7.5, 8.5, 9.5, 10.5]
        #              slow sma [6.5, 7.0, 7.5, 8.5, 9.5]
        assert out == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    def test_warmup_is_flat_not_nan(self):
        df = frame(range(1, 6))
        out = SmaCross(fast=2, slow=4).stance_series(df)
        assert out.iloc[:3].tolist() == [0, 0, 0]
        assert not out.isna().any()

    def test_rationale_mentions_smas(self):
        df = frame(range(1, 20))
        text = SmaCross(fast=2, slow=4).rationale(df)
        assert "SMA(2)" in text and "SMA(4)" in text and "uptrend" in text

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="fast"):
            SmaCross(fast=50, slow=20)


class TestRocMomentum:
    def test_stance_is_sign_of_lookback_return(self):
        df = frame([1, 2, 3, 2, 1, 0.5, 2, 3])
        out = RocMomentum(lookback=2).stance_series(df).tolist()
        # close[t] > close[t-2] ? (first two bars are warmup)
        assert out == [0, 0, 1, 0, 0, 0, 1, 1]

    def test_rationale(self):
        df = frame([1, 2, 3, 4])
        text = RocMomentum(lookback=2).rationale(df)
        assert "2d return" in text and "momentum" in text


class TestRsiMeanReversion:
    def test_hysteresis_enter_hold_exit(self):
        # steep drop pushes RSI(3) to ~0 (enter), recovery lifts it past 60 (exit)
        closes = [100, 100, 100, 90, 80, 70, 71, 74, 78, 85, 95, 100]
        df = frame(closes)
        strat = RsiMeanReversion(period=3, enter_below=30, exit_above=60)
        values = rsi(df["close"], 3)
        out = strat.stance_series(df)

        entered = out[out == 1]
        assert not entered.empty, f"never entered; rsi={values.round(1).tolist()}"
        first_entry = entered.index[0]
        assert values[first_entry] < 30
        # once entered, stays long while RSI is between the bands
        exited = out[(out.index > first_entry) & (out == 0)]
        assert not exited.empty, "never exited"
        first_exit = exited.index[0]
        assert values[first_exit] > 60
        between = out.loc[first_entry:first_exit - 1]
        assert (between == 1).all()

    def test_no_nan_and_warmup_flat(self):
        df = frame([100] * 3 + [50] * 3)
        out = RsiMeanReversion(period=3).stance_series(df)
        assert not out.isna().any()
        assert out.iloc[0] == 0

    def test_invalid_bands(self):
        with pytest.raises(ValueError, match="enter_below"):
            RsiMeanReversion(enter_below=60, exit_above=50)


def test_registry_builds_from_config(fixture_config):
    strategies = build_strategies(fixture_config)
    names = [s.name for s in strategies]
    assert names == ["trend_sma", "momentum_roc", "meanrev_rsi"]
    weights = {s.name: s.weight for s in strategies}
    assert weights["meanrev_rsi"] == 0.5
    trend = strategies[0]
    assert (trend.fast, trend.slow) == (20, 100)


def test_stances_only_use_past_data(fixture_config):
    """Changing the last bar must not change any earlier stance."""
    from daily_signals.data.cache import load

    df = load(fixture_config, "BTC-USD")
    for strategy in build_strategies(fixture_config):
        base = strategy.stance_series(df)
        tampered = df.copy()
        tampered.loc[tampered.index[-1], "close"] *= 5
        after = strategy.stance_series(tampered)
        assert base.iloc[:-1].equals(after.iloc[:-1]), strategy.name
