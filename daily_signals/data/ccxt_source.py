"""Fallback data source via ccxt public endpoints (no API key).

ccxt is not part of the light requirements; it is imported lazily so the
pipeline only needs it if the Coinbase source fails AND this fallback is
actually invoked (or ``data.source: ccxt`` is configured).
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from daily_signals.data.base import DataError, normalize_frame, utc_today


class CcxtSource:
    def __init__(self, exchange_id: str = "kraken") -> None:
        self.exchange_id = exchange_id

    def fetch(self, symbol: str, start: Optional[dt.date] = None) -> pd.DataFrame:
        try:
            import ccxt  # lazy: optional dependency
        except ImportError as exc:
            raise DataError(
                "ccxt is not installed — `pip install ccxt` to enable the "
                f"{self.exchange_id} fallback data source"
            ) from exc

        try:
            exchange = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        except AttributeError as exc:
            raise DataError(f"unknown ccxt exchange id: {self.exchange_id}") from exc

        market = symbol.replace("-", "/")  # BTC-USD -> BTC/USD
        today = utc_today()
        start = start or today - dt.timedelta(days=400)
        since = int(dt.datetime(start.year, start.month, start.day,
                                tzinfo=dt.timezone.utc).timestamp() * 1000)

        rows = []
        while True:
            batch = exchange.fetch_ohlcv(market, timeframe="1d", since=since, limit=720)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= since and len(batch) > 1:
                break  # no forward progress; avoid infinite loop
            since = last_ts + 24 * 3600 * 1000
            if len(batch) < 2 or dt.datetime.fromtimestamp(
                last_ts / 1000, dt.timezone.utc
            ).date() >= today:
                break

        if not rows:
            raise DataError(f"{symbol}: {self.exchange_id} returned no candles")
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["time"], unit="ms")
        return normalize_frame(df, symbol, today=today)
