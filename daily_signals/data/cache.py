"""Local OHLCV cache: one CSV per pair, incrementally refreshed.

The cache is the single place the rest of the pipeline reads bars from.
Refresh = read cache -> fetch from a few days before the last cached bar
(overlap absorbs upstream corrections) -> merge, dedupe, validate -> atomic
write. ``data.source: fixture`` redirects reads to the committed test
fixtures and never touches the network.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from daily_signals.data.base import (
    CANONICAL_COLUMNS,
    DataError,
    gap_report,
    normalize_frame,
    utc_today,
    validate_frame,
)

REFRESH_OVERLAP_DAYS = 3


def cache_path(cfg, symbol: str) -> Path:
    base = Path(cfg.data.fixture_dir if cfg.data.source == "fixture" else cfg.data.cache_dir)
    return base / f"{symbol}_1d.csv"


def read_frame(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_frame(df, symbol, today=utc_today() + dt.timedelta(days=1))
    # (today+1 keeps every stored bar; fixtures end long before "today")
    return validate_frame(df, symbol)


def write_frame(path: Path, df: pd.DataFrame) -> None:
    """Atomic write: temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            df.to_csv(fh, index=False, columns=CANONICAL_COLUMNS)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load(cfg, symbol: str) -> pd.DataFrame:
    path = cache_path(cfg, symbol)
    if not path.exists():
        raise DataError(
            f"{symbol}: no cached data at {path}. Run 'daily-signals fetch' first."
        )
    return read_frame(path, symbol)


def load_all(cfg) -> Dict[str, pd.DataFrame]:
    return {asset.symbol: load(cfg, asset.symbol) for asset in cfg.assets}


def _build_source(cfg, fallback: bool = False):
    if fallback or cfg.data.source == "ccxt":
        from daily_signals.data.ccxt_source import CcxtSource

        return CcxtSource(exchange_id=cfg.data.ccxt_exchange)
    from daily_signals.data.coinbase import CoinbaseSource

    return CoinbaseSource()


def _initial_start(cfg) -> dt.date:
    backtest_start = dt.date.fromisoformat(cfg.backtest.start)
    history_start = utc_today() - dt.timedelta(days=cfg.data.min_history_days)
    return min(backtest_start, history_start)


def refresh(cfg, symbol: str, source=None) -> pd.DataFrame:
    """Fetch new bars for ``symbol``, merge into the cache, return the result."""
    if cfg.data.source == "fixture":
        return load(cfg, symbol)

    path = cache_path(cfg, symbol)
    cached: Optional[pd.DataFrame] = None
    if path.exists():
        cached = read_frame(path, symbol)
        start = cached["date"].iloc[-1].date() - dt.timedelta(days=REFRESH_OVERLAP_DAYS)
    else:
        start = _initial_start(cfg)

    if source is None:
        try:
            fresh = _build_source(cfg).fetch(symbol, start=start)
        except Exception as primary_exc:
            if cfg.data.source == "ccxt":
                raise
            print(
                f"warning: {symbol}: primary source failed ({primary_exc}); "
                f"trying ccxt/{cfg.data.ccxt_exchange}",
                file=sys.stderr,
            )
            fresh = _build_source(cfg, fallback=True).fetch(symbol, start=start)
    else:
        fresh = source.fetch(symbol, start=start)

    if cached is not None:
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = (
            merged.sort_values("date")
            .drop_duplicates("date", keep="last")  # fresh rows win on overlap
            .reset_index(drop=True)
        )
    else:
        merged = fresh

    merged = validate_frame(merged, symbol)
    gaps = gap_report(merged)
    if gaps:
        preview = ", ".join(str(g) for g in gaps[:5])
        print(
            f"warning: {symbol}: {len(gaps)} missing calendar day(s) in history "
            f"(e.g. {preview})",
            file=sys.stderr,
        )
    write_frame(path, merged)
    return merged


def refresh_all(cfg) -> Dict[str, pd.DataFrame]:
    return {asset.symbol: refresh(cfg, asset.symbol) for asset in cfg.assets}
