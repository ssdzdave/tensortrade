"""Scale-invariant daily features shared by RL training and inference.

Mirrors the feature family that worked in examples/training/train_best.py
(returns, RSI, SMA trend, volatility, Bollinger position) with windows in
days instead of hours. Warmup rows are dropped, never back-filled — bfill
would leak future data into the training set.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

#: longest lookback used below; callers drop this many warmup rows
WARMUP_ROWS = 60


def add_daily_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return (frame with feature columns, feature column names)."""
    df = df.copy()

    for period in (1, 5, 10, 20):
        df[f"ret_{period}d"] = np.tanh(df["close"].pct_change(period) * 10)

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_norm"] = (100 - (100 / (1 + rs)) - 50) / 50

    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    df["trend_20"] = np.tanh((df["close"] - sma20) / sma20 * 10)
    df["trend_50"] = np.tanh((df["close"] - sma50) / sma50 * 10)
    df["trend_strength"] = np.tanh((sma20 - sma50) / sma50 * 20)

    vol = df["close"].rolling(20).std() / df["close"]
    df["vol_norm"] = np.tanh(
        (vol - vol.rolling(60).mean()) / (vol.rolling(60).std() + 1e-10)
    )

    df["volume_ratio"] = np.log1p(
        df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    )

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_pos"] = (
        (df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)
    ).clip(0, 1)

    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d", "rsi_norm",
        "trend_20", "trend_50", "trend_strength", "vol_norm",
        "volume_ratio", "bb_pos",
    ]
    return df, feature_cols


def featurize(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Features + warmup rows dropped — ready for an env or a replay."""
    out, cols = add_daily_features(df)
    out = out.dropna(subset=cols).reset_index(drop=True)
    return out, cols
