"""Metrics pinned against hand-computed values on tiny fixed series."""

import math

import numpy as np
import pandas as pd
import pytest

from daily_signals import metrics


def test_cagr_constant_daily_return():
    r = pd.Series([0.01] * 365)
    expected = 1.01 ** 365 - 1  # exactly one year at 365 ppy
    assert metrics.cagr(r) == pytest.approx(expected)


def test_cagr_half_year_annualizes():
    # +10% total over half a year -> (1.1)^2 - 1 annualized
    n = 365 // 2
    daily = 1.1 ** (1 / n) - 1
    r = pd.Series([daily] * n)
    assert metrics.cagr(r) == pytest.approx(1.1 ** (365 / n) - 1, rel=1e-6)


def test_sharpe_hand_computed():
    r = pd.Series([0.02, -0.01, 0.02, -0.01])  # mean=.005, sd(ddof=1)=sqrt(3e-4)
    expected = 0.005 / math.sqrt(0.0003) * math.sqrt(365)
    assert metrics.sharpe(r) == pytest.approx(expected)


def test_sharpe_zero_variance_is_nan():
    assert math.isnan(metrics.sharpe(pd.Series([0.01] * 10)))


def test_sortino_all_positive_is_inf():
    assert metrics.sortino(pd.Series([0.01, 0.02, 0.03])) == float("inf")


def test_sortino_hand_computed():
    r = pd.Series([0.10, -0.10, 0.10, -0.10])
    downside_dev = math.sqrt((0.10 ** 2 + 0.10 ** 2) / 4)
    expected = 0.0 / downside_dev  # mean is zero
    assert metrics.sortino(r) == pytest.approx(expected)


def test_max_drawdown_hand_computed():
    equity = pd.Series([100, 110, 99, 105, 120, 90])
    # 110 -> 99 is -10%; 120 -> 90 is -25% -> MDD = -25%
    assert metrics.max_drawdown(equity) == pytest.approx(-0.25)


def test_max_drawdown_monotonic_up_is_zero():
    assert metrics.max_drawdown(pd.Series([1, 2, 3])) == 0.0


def test_trade_list_and_win_rate():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    pos = pd.Series([0, 1, 1, 0, 1, 0], index=idx)
    ret = pd.Series([0.0, 0.10, 0.10, 0.0, -0.05, 0.0], index=idx)
    trades = metrics.trade_list(pos, ret)
    assert len(trades) == 2
    assert trades.loc[0, "return"] == pytest.approx(1.1 * 1.1 - 1)
    assert trades.loc[0, "entry"] == idx[1] and trades.loc[0, "exit"] == idx[2]
    assert trades.loc[1, "return"] == pytest.approx(-0.05)
    assert metrics.win_rate(trades) == pytest.approx(0.5)


def test_trade_list_open_trade_at_end():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    trades = metrics.trade_list(
        pd.Series([0, 1, 1], index=idx), pd.Series([0.0, 0.1, 0.1], index=idx)
    )
    assert len(trades) == 1
    assert trades.loc[0, "exit"] == idx[-1]


def test_exposure_and_trades_per_month():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    pos = pd.Series([1] * 15 + [0] * 15, index=idx)
    assert metrics.exposure(pos) == pytest.approx(0.5)
    # one position change over ~1 month
    assert metrics.trades_per_month(pos) == pytest.approx(
        1 / (30 / 365 * 12), rel=1e-9
    )


def test_summarize_contains_all_fields_and_vs_buy_hold():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.0, 0.09, 0.10, -0.01, 0.09], index=idx)
    positions = pd.Series([0, 1, 1, 0, 1], index=idx)
    bench = pd.Series([0.0, 0.10, 0.10, -0.10, 0.10], index=idx)
    row = metrics.summarize(returns, positions, benchmark_returns=bench)
    for field in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                  "win_rate", "num_trades", "trades_per_month", "exposure",
                  "vs_buy_hold"):
        assert field in row, field
    strat_total = (1 + returns).prod() - 1
    bench_total = (1 + bench).prod() - 1
    assert row["total_return"] == pytest.approx(strat_total)
    assert row["vs_buy_hold"] == pytest.approx(strat_total - bench_total)
