"""Time-series momentum: long while the N-day return is positive."""

from __future__ import annotations

import pandas as pd

from daily_signals.strategies.base import Strategy


class RocMomentum(Strategy):
    name = "momentum_roc"

    def __init__(self, weight: float = 1.0, lookback: int = 90) -> None:
        super().__init__(weight=weight, lookback=lookback)
        self.lookback = lookback

    def _roc(self, df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(self.lookback)

    def stance_series(self, df: pd.DataFrame) -> pd.Series:
        roc = self._roc(df)
        return self._finalize((roc > 0).astype(float).where(roc.notna()))

    def rationale(self, df: pd.DataFrame) -> str:
        roc = self._roc(df).iloc[-1]
        if pd.isna(roc):
            return f"warming up: needs {self.lookback} bars of history"
        sign = "positive" if roc > 0 else "non-positive"
        return f"{self.lookback}d return {roc:+.1%} — momentum {sign}"
