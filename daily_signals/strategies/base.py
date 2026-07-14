"""Strategy contract.

A strategy maps a canonical daily OHLCV frame to a **stance series**:
one int per bar, ``1`` = LONG, ``0`` = FLAT, where the value at row ``t``
uses only information up to and including close(t). Warmup bars are flat.

The same series serves the backtest (full history) and the daily signal
(its last value), so what clients receive is exactly what was tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    #: registry key, set by subclasses (e.g. "trend_sma")
    name: str = ""

    def __init__(self, weight: float = 1.0, **params) -> None:
        self.weight = weight
        self.params = params

    @abstractmethod
    def stance_series(self, df: pd.DataFrame) -> pd.Series:
        """Return an int 0/1 series aligned to ``df`` rows."""

    @abstractmethod
    def rationale(self, df: pd.DataFrame) -> str:
        """One line explaining the latest stance, for client reports."""

    def latest_stance(self, df: pd.DataFrame) -> int:
        return int(self.stance_series(df).iloc[-1])

    @staticmethod
    def _finalize(raw: pd.Series) -> pd.Series:
        """NaN warmup -> flat, everything to plain ints."""
        return raw.fillna(0).astype(int)
