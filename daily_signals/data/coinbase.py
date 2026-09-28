"""Primary data source: Coinbase Exchange public market data.

``GET https://api.exchange.coinbase.com/products/{product}/candles`` needs no
API key. Candles arrive as ``[time, low, high, open, close, volume]`` rows,
max 300 per request, so longer histories are fetched in date windows.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Optional

import pandas as pd
import requests

from daily_signals.data.base import DataError, normalize_frame, utc_today

BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 86400  # daily
MAX_CANDLES_PER_REQUEST = 300
DEFAULT_TIMEOUT = 30
RETRIES = 3


class CoinbaseSource:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "daily-signals/0.1")

    def fetch(self, symbol: str, start: Optional[dt.date] = None) -> pd.DataFrame:
        today = utc_today()
        start = start or today - dt.timedelta(days=400)
        frames = []
        window_start = start
        while window_start <= today:
            window_end = min(
                window_start + dt.timedelta(days=MAX_CANDLES_PER_REQUEST - 1), today
            )
            rows = self._request_candles(symbol, window_start, window_end)
            if rows:
                frames.append(pd.DataFrame(
                    rows, columns=["time", "low", "high", "open", "close", "volume"]
                ))
            window_start = window_end + dt.timedelta(days=1)
            time.sleep(0.15)  # stay well under the public rate limit

        if not frames:
            raise DataError(f"{symbol}: coinbase returned no candles since {start}")
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["time"], unit="s")
        return normalize_frame(df, symbol, today=today)

    def _request_candles(self, symbol: str, start: dt.date, end: dt.date) -> list:
        url = f"{BASE_URL}/products/{symbol}/candles"
        params = {
            "granularity": GRANULARITY,
            "start": f"{start.isoformat()}T00:00:00Z",
            # end is inclusive of the bucket starting at end 00:00 UTC
            "end": f"{end.isoformat()}T00:00:00Z",
        }
        last_error: Optional[Exception] = None
        for attempt in range(RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
                if resp.status_code == 404:
                    raise DataError(
                        f"{symbol}: unknown Coinbase product (HTTP 404) — "
                        "check the symbol, e.g. BTC-USD"
                    )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list):
                    raise DataError(f"{symbol}: unexpected candle payload: {payload!r:.200}")
                return payload
            except DataError:
                raise
            except Exception as exc:  # connection errors, 5xx, timeouts
                last_error = exc
                time.sleep(2 ** attempt)
        raise DataError(f"{symbol}: coinbase fetch failed after {RETRIES} attempts: {last_error}")
