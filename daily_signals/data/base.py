"""Data source contract and frame validation.

Every source returns the same canonical frame so the rest of the pipeline
never cares where bars came from:

* columns: ``date`` (naive datetime64, midnight UTC), ``open``, ``high``,
  ``low``, ``close``, ``volume`` (floats)
* strictly ascending, unique dates
* **only fully closed UTC days** — the in-progress bar is always dropped
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Protocol

import pandas as pd

CANONICAL_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """Raised when a source or cache produces an invalid frame."""


class OHLCVSource(Protocol):
    def fetch(self, symbol: str, start: Optional[dt.date] = None) -> pd.DataFrame:
        """Return canonical daily bars for ``symbol`` from ``start`` onward."""
        ...


def utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def normalize_frame(df: pd.DataFrame, symbol: str,
                    today: Optional[dt.date] = None) -> pd.DataFrame:
    """Sort, dedupe, coerce dtypes and drop the in-progress UTC bar."""
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = df[CANONICAL_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for col in CANONICAL_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    today = today or utc_today()
    df = df[df["date"] < pd.Timestamp(today)]
    return df.reset_index(drop=True)


def validate_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Fail loudly on malformed data rather than signal on garbage."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")
    if df.empty:
        raise DataError(f"{symbol}: no bars")
    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(), "date"].head(3).tolist()
        raise DataError(f"{symbol}: duplicate dates, e.g. {dupes}")
    if not df["date"].is_monotonic_increasing:
        raise DataError(f"{symbol}: dates are not ascending")
    if df["close"].isna().any():
        n = int(df["close"].isna().sum())
        raise DataError(f"{symbol}: {n} NaN close prices")
    if (df["close"] <= 0).any():
        raise DataError(f"{symbol}: non-positive close prices")
    return df


def gap_report(df: pd.DataFrame) -> list:
    """Return missing calendar dates (crypto trades 24/7, so gaps are data
    issues worth surfacing — but not fatal)."""
    if len(df) < 2:
        return []
    full = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D")
    have = set(df["date"])
    return [d.date() for d in full if d not in have]
