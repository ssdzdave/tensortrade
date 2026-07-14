"""RLVoice: turn a trained per-asset PPO checkpoint into an ensemble voice.

Inference restores the *policy* sub-checkpoint (``Policy.from_checkpoint``)
rather than the whole algorithm, so no env workers or Ray session spin up
inside the daily job. The policy replays deterministically over the recent
feature window; the final action is today's stance and the replayed series
feeds the shared backtester so the RL row of the tearsheet is measured by
the exact engine the rule voices use.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from daily_signals.rl.features import featurize

REPLAY_BARS = 250


class RLVoice:
    def __init__(self, cfg) -> None:
        if importlib.util.find_spec("ray") is None:
            raise RuntimeError(
                "ray is not installed — `pip install -e \".[rl]\"` to enable "
                "the RL voice"
            )
        self.cfg = cfg
        self.models_dir = Path(cfg.rl.models_dir)
        self._cache: Dict[str, tuple] = {}

    def _load(self, symbol: str):
        if symbol in self._cache:
            return self._cache[symbol]
        manifest_path = self.models_dir / symbol / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"no trained model for {symbol} (expected {manifest_path}; "
                f"run `python -m daily_signals.rl.train_daily --asset {symbol}`)"
            )
        manifest = json.loads(manifest_path.read_text())
        from ray.rllib.policy.policy import Policy

        policy_path = self.models_dir / symbol / manifest["policy_relpath"]
        if not policy_path.exists():
            raise RuntimeError(f"{symbol}: checkpoint missing at {policy_path}")
        policy = Policy.from_checkpoint(str(policy_path))
        if isinstance(policy, dict):  # algo-level checkpoint returns a mapping
            policy = policy.get("default_policy") or next(iter(policy.values()))
        self._cache[symbol] = (policy, manifest)
        return self._cache[symbol]

    def stance_series(self, symbol: str, df: pd.DataFrame,
                      replay_bars: int = REPLAY_BARS) -> pd.Series:
        """Deterministic stance history over the recent window (date-indexed)."""
        from daily_signals.rl.envs import create_frame_env

        policy, manifest = self._load(symbol)
        window_size = int(manifest["window_size"])

        frame, feature_cols = featurize(df)
        if manifest["feature_cols"] != feature_cols:
            raise RuntimeError(
                f"{symbol}: feature set changed since training "
                f"({manifest['feature_cols']} vs {feature_cols}) — retrain"
            )
        frame = frame.tail(replay_bars + window_size).reset_index(drop=True)
        if len(frame) <= window_size + 1:
            raise RuntimeError(f"{symbol}: not enough bars to replay")

        env = create_frame_env({
            "frame": frame,
            "feature_cols": feature_cols,
            "window_size": window_size,
            "commission": 0.0,          # replay observes, it doesn't pay fees
            "max_allowed_loss": 0.99,   # never stop out mid-replay
            "symbol": symbol,
        })
        obs, _ = env.reset()
        actions = []
        done = truncated = False
        while not done and not truncated:
            action, _, _ = policy.compute_single_action(obs, explore=False)
            actions.append(int(action))
            obs, _, done, truncated, _ = env.step(action)

        # action k reacts to the observation ending at bar (window_size-1+k)
        start = window_size - 1
        index = frame["date"].iloc[start:start + len(actions)]
        return pd.Series(actions[:len(index)], index=pd.DatetimeIndex(index))

    def stance(self, symbol: str, df: pd.DataFrame) -> Tuple[int, str]:
        """(today's stance, one-line rationale) for the ensemble."""
        series = self.stance_series(symbol, df)
        _, manifest = self._load(symbol)
        stance = int(series.iloc[-1])
        held = "holding the asset" if stance == 1 else "sitting in cash"
        flips = int(series.diff().abs().fillna(0).sum())
        rationale = (
            f"PPO policy ({manifest.get('trained_at', '?')[:10]}, "
            f"val P&L ${manifest.get('val_pnl', 0):+,.0f}) is {held}; "
            f"{flips} flips in the last {len(series)} bars"
        )
        return stance, rationale
