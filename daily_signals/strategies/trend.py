"""Trend following: long while the fast SMA is above the slow SMA."""

from __future__ import annotations

import pandas as pd

from daily_signals.strategies.base import Strategy


class SmaCross(Strategy):
    name = "trend_sma"

    def __init__(self, weight: float = 1.0, fast: int = 20, slow: int = 100) -> None:
        if fast >= slow:
            raise ValueError(f"trend_sma: fast ({fast}) must be < slow ({slow})")
        super().__init__(weight=weight, fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    def _smas(self, df: pd.DataFrame):
        close = df["close"]
        return close.rolling(self.fast).mean(), close.rolling(self.slow).mean()

    def stance_series(self, df: pd.DataFrame) -> pd.Series:
        fast_sma, slow_sma = self._smas(df)
        return self._finalize((fast_sma > slow_sma).astype(float).where(slow_sma.notna()))

    def rationale(self, df: pd.DataFrame) -> str:
        fast_sma, slow_sma = self._smas(df)
        f, s = fast_sma.iloc[-1], slow_sma.iloc[-1]
        if pd.isna(s):
            return f"warming up: needs {self.slow} bars of history"
        direction = "above" if f > s else "below"
        verdict = "uptrend" if f > s else "no uptrend"
        return (
            f"SMA({self.fast}) {f:,.2f} is {direction} SMA({self.slow}) {s:,.2f}"
            f" — {verdict}"
        )
