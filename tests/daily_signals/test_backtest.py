"""Backtester tests: hand-computed 5-bar arithmetic, no-lookahead, tearsheet."""

import numpy as np
import pandas as pd
import pytest

from daily_signals.backtest import (
    BUY_HOLD,
    ENSEMBLE,
    PORTFOLIO,
    equity_curves,
    render_tearsheet_text,
    run_backtest,
    tearsheet,
)
from daily_signals.data.cache import load_all


def series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_five_bar_hand_computed_arithmetic():
    prices = series([100.0, 110.0, 121.0, 108.9, 119.79])
    stances = series([1, 1, 0, 1, 1]).astype(int)
    result = run_backtest(prices, stances, commission_bps=100, initial_cash=10_000)

    # pos = stance.shift(1) = [0,1,1,0,1]; market ret = [0,.1,.1,-.1,.1]
    # costs at position changes (1% each): bars 1, 3, 4
    expected_returns = [0.0, 0.09, 0.10, -0.01, 0.09]
    assert result.returns.tolist() == pytest.approx(expected_returns)
    assert result.positions.tolist() == [0, 1, 1, 0, 1]
    assert result.num_trades == 3
    expected_final = 10_000 * 1.09 * 1.10 * 0.99 * 1.09
    assert result.equity.iloc[-1] == pytest.approx(expected_final)  # 12,938.409


def test_no_lookahead_stance_acts_next_bar():
    """A stance built from same-bar returns must NOT capture those returns."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.02, 200)
    prices = series(100 * np.exp(np.cumsum(rets)))
    market = prices.pct_change().fillna(0)
    cheat = (market > 0).astype(int)  # "knows" today's move at today's close

    result = run_backtest(prices, cheat, commission_bps=0)

    # independent recomputation with the correct 1-bar lag
    lagged = cheat.shift(1).fillna(0).astype(int)
    expected = lagged * market
    assert result.returns.tolist() == pytest.approx(expected.tolist())

    # if the engine leaked, it would have earned every positive bar
    leaked_total = float((1 + market[cheat == 1]).prod())
    achieved = float((1 + result.returns).prod())
    assert achieved < leaked_total


def test_start_slices_window_but_keeps_prior_position():
    prices = series([100, 110, 121, 133.1, 146.41])
    stances = series([1, 1, 1, 1, 1]).astype(int)
    result = run_backtest(prices, stances, commission_bps=100,
                          start=str(prices.index[2].date()))
    # entry cost was charged before the window -> not visible here
    assert result.returns.tolist() == pytest.approx([0.10, 0.10, 0.10])
    assert (result.positions == 1).all()


def test_buy_hold_pays_entry_commission_once():
    prices = series([100, 110, 121])
    always = series([1, 1, 1]).astype(int)
    result = run_backtest(prices, always, commission_bps=100)
    assert result.returns.tolist() == pytest.approx([0.0, 0.09, 0.10])
    assert result.num_trades == 1


def test_tearsheet_shape_and_sanity(fixture_config, no_network):
    data = load_all(fixture_config)
    sheet = tearsheet(fixture_config, data)

    voices = {"trend_sma", "momentum_roc", "meanrev_rsi", ENSEMBLE, BUY_HOLD}
    for asset in ("BTC-USD", "ETH-USD", "SOL-USD"):
        assert set(sheet.loc[sheet["asset"] == asset, "voice"]) == voices
    assert set(sheet.loc[sheet["asset"] == PORTFOLIO, "voice"]) == {ENSEMBLE, BUY_HOLD}

    # buy & hold measured against itself is zero edge
    bh = sheet[(sheet["voice"] == BUY_HOLD)]
    assert bh["vs_buy_hold"].abs().max() == pytest.approx(0.0)
    # every strategy trades less than ~one flip per day (sanity bound)
    assert sheet["trades_per_month"].max() < 31

    text = render_tearsheet_text(sheet)
    assert "PORTFOLIO" in text and "%" in text


def test_equity_curves_columns(fixture_config, no_network):
    data = load_all(fixture_config)
    curves = equity_curves(fixture_config, data)
    assert f"{PORTFOLIO} ensemble" in curves.columns
    assert f"{PORTFOLIO} buy&hold" in curves.columns
    assert "BTC-USD ensemble" in curves.columns
    assert curves.notna().all().all()
