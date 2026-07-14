"""Mean reversion: buy oversold RSI, exit once it normalizes.

Stateful hysteresis — enter when RSI drops below ``enter_below``, stay long
until RSI rises above ``exit_above``. The RSI uses simple rolling means of
gains/losses, matching the feature construction used elsewhere in this repo
(see examples/training/train_best.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from daily_signals.strategies.base import Strategy


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


class RsiMeanReversion(Strategy):
    name = "meanrev_rsi"

    def __init__(self, weight: float = 1.0, period: int = 14,
                 enter_below: float = 30, exit_above: float = 55) -> None:
        if enter_below >= exit_above:
            raise ValueError(
                f"meanrev_rsi: enter_below ({enter_below}) must be < "
                f"exit_above ({exit_above})"
            )
        super().__init__(weight=weight, period=period,
                         enter_below=enter_below, exit_above=exit_above)
        self.period = period
        self.enter_below = enter_below
        self.exit_above = exit_above

    def stance_series(self, df: pd.DataFrame) -> pd.Series:
        values = rsi(df["close"], self.period).to_numpy()
        stance = np.zeros(len(values), dtype=int)
        holding = False
        for i, value in enumerate(values):
            if np.isnan(value):
                stance[i] = 0
                continue
            if not holding and value < self.enter_below:
                holding = True
            elif holding and value > self.exit_above:
                holding = False
            stance[i] = int(holding)
        return pd.Series(stance, index=df.index)

    def rationale(self, df: pd.DataFrame) -> str:
        value = rsi(df["close"], self.period).iloc[-1]
        if pd.isna(value):
            return f"warming up: needs {self.period + 1} bars of history"
        holding = bool(self.stance_series(df).iloc[-1])
        if holding:
            return (
                f"RSI({self.period}) {value:.1f} — bought oversold "
                f"(<{self.enter_below:g}), holding until >{self.exit_above:g}"
            )
        if value < self.enter_below:
            return f"RSI({self.period}) {value:.1f} < {self.enter_below:g} — oversold"
        return (
            f"RSI({self.period}) {value:.1f} — not oversold "
            f"(entry <{self.enter_below:g})"
        )
