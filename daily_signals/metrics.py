"""Performance metrics for daily return/position series.

Self-contained (quantstats has build issues on Python 3.12+, and Sharpe in
this repo previously existed only as an RL reward). All functions take daily
series; crypto trades every day, so ``periods_per_year`` defaults to 365.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

PPY = 365  # periods per year for a 24/7 market


def cagr(returns: pd.Series, ppy: int = PPY) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    growth = float((1 + returns).prod())
    if growth <= 0:
        return -1.0
    years = len(returns) / ppy
    if years == 0:
        return 0.0
    return growth ** (1 / years) - 1


def sharpe(returns: pd.Series, rf: float = 0.0, ppy: int = PPY) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf / ppy
    std = float(excess.std(ddof=1))
    if std < 1e-12:  # constant series (float noise included) has no Sharpe
        return float("nan")
    return float(excess.mean()) / std * math.sqrt(ppy)


def sortino(returns: pd.Series, rf: float = 0.0, ppy: int = PPY) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf / ppy
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    downside_dev = math.sqrt(float((downside ** 2).sum()) / len(excess))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean()) / downside_dev * math.sqrt(ppy)


def max_drawdown(equity: pd.Series) -> float:
    """Deepest peak-to-trough decline, as a negative fraction."""
    equity = equity.dropna()
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    return equity / equity.cummax() - 1


def trade_list(positions: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Segment held periods into trades with compounded net returns.

    ``positions[t]`` is the position that earned ``returns[t]``; a trade is a
    maximal run of ``positions == 1``.
    """
    positions = positions.fillna(0).astype(int)
    trades = []
    entry_idx = None
    for i in range(len(positions)):
        holding = positions.iloc[i] == 1
        if holding and entry_idx is None:
            entry_idx = i
        elif not holding and entry_idx is not None:
            segment = returns.iloc[entry_idx:i]
            trades.append(
                {
                    "entry": positions.index[entry_idx],
                    "exit": positions.index[i - 1],
                    "return": float((1 + segment.fillna(0)).prod() - 1),
                }
            )
            entry_idx = None
    if entry_idx is not None:  # still open at the end
        segment = returns.iloc[entry_idx:]
        trades.append(
            {
                "entry": positions.index[entry_idx],
                "exit": positions.index[-1],
                "return": float((1 + segment.fillna(0)).prod() - 1),
            }
        )
    return pd.DataFrame(trades, columns=["entry", "exit", "return"])


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    return float((trades["return"] > 0).mean())


def exposure(positions: pd.Series) -> float:
    positions = positions.fillna(0)
    if positions.empty:
        return 0.0
    return float((positions != 0).mean())


def trades_per_month(positions: pd.Series, ppy: int = PPY) -> float:
    """Position changes per month — the commission-drag number to watch."""
    positions = positions.fillna(0).astype(int)
    if len(positions) < 2:
        return 0.0
    changes = int(positions.diff().abs().fillna(0).sum())
    months = len(positions) / ppy * 12
    return changes / months if months > 0 else 0.0


def summarize(returns: pd.Series, positions: pd.Series,
              benchmark_returns: Optional[pd.Series] = None) -> dict:
    """One tearsheet row for a strategy's net daily returns + positions."""
    equity = (1 + returns.fillna(0)).cumprod()
    trades = trade_list(positions, returns)
    row = {
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "cagr": cagr(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(trades),
        "num_trades": len(trades),
        "trades_per_month": trades_per_month(positions),
        "exposure": exposure(positions),
    }
    if benchmark_returns is not None:
        bench_equity = (1 + benchmark_returns.fillna(0)).cumprod()
        row["vs_buy_hold"] = row["total_return"] - float(bench_equity.iloc[-1] - 1)
    return row
