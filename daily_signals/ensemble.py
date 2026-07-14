"""Combine strategy voices into one explainable stance per asset.

Weighted vote: score = sum(weight_i * stance_i) / sum(weight_i) over the
voices that are *present* (an unavailable RL voice simply drops out of the
denominator). Score above the threshold means LONG, below means FLAT, and an
exact tie keeps yesterday's stance — statefulness that damps flip-flopping,
the failure mode documented in docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Voice:
    name: str
    stance: int  # 1 = LONG, 0 = FLAT
    weight: float
    rationale: str
    available: bool = True


@dataclass
class EnsembleResult:
    stance: int
    score: float
    prev_stance: int
    changed: bool
    n_long: int
    n_voices: int
    voices: List[Voice] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "LONG" if self.stance == 1 else "FLAT"

    def vote_summary(self) -> str:
        tied = f"{self.n_long} of {self.n_voices} voices long"
        if self.score == 0.5 and self.n_voices:
            return f"{tied} (tie — keeping previous stance)"
        return f"{tied} (weighted score {self.score:.2f})"


def combine_voices(voices: List[Voice], prev_stance: int = 0,
                   long_threshold: float = 0.5) -> EnsembleResult:
    present = [v for v in voices if v.available]
    total_weight = sum(v.weight for v in present)
    if total_weight <= 0:
        stance, score = prev_stance, 0.5
    else:
        score = sum(v.weight * v.stance for v in present) / total_weight
        if score > long_threshold:
            stance = 1
        elif score < long_threshold:
            stance = 0
        else:
            stance = prev_stance
    return EnsembleResult(
        stance=stance,
        score=score,
        prev_stance=prev_stance,
        changed=stance != prev_stance,
        n_long=sum(1 for v in present if v.stance == 1),
        n_voices=len(present),
        voices=list(voices),
    )


def ensemble_series(voice_series: Dict[str, pd.Series],
                    weights: Dict[str, float],
                    long_threshold: float = 0.5,
                    initial: int = 0) -> pd.Series:
    """Apply the same vote bar-by-bar over aligned stance histories.

    Used by the backtest so the tearsheet's "ensemble" row is produced by the
    identical rule the daily signal uses.
    """
    if not voice_series:
        raise ValueError("ensemble_series needs at least one voice")
    frame = pd.DataFrame(voice_series)
    weight_vector = pd.Series({name: weights[name] for name in frame.columns})
    total = float(weight_vector.sum())
    if total <= 0:
        raise ValueError("ensemble weights must sum to > 0")
    score = frame.mul(weight_vector, axis=1).sum(axis=1) / total

    values = np.empty(len(frame), dtype=int)
    prev = initial
    long_mask = (score > long_threshold).to_numpy()
    flat_mask = (score < long_threshold).to_numpy()
    # ties are rare; a simple pass keeps the logic identical to combine_voices
    for i in range(len(frame)):
        if long_mask[i]:
            prev = 1
        elif flat_mask[i]:
            prev = 0
        values[i] = prev
    return pd.Series(values, index=frame.index)


def compute_voices(cfg, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Voice]]:
    """Latest Voice per strategy (and optional RL) for every configured asset."""
    from daily_signals.strategies import build_strategies

    strategies = build_strategies(cfg)

    rl = None
    rl_error: Optional[str] = None
    if cfg.rl.enabled:
        try:
            from daily_signals.rl.voice import RLVoice

            rl = RLVoice(cfg)
        except Exception as exc:  # ray missing, no checkpoint, etc.
            rl_error = str(exc)
            print(f"warning: RL voice unavailable: {exc}", file=sys.stderr)

    out: Dict[str, List[Voice]] = {}
    for asset in cfg.assets:
        df = data[asset.symbol]
        voices = [
            Voice(
                name=s.name,
                stance=s.latest_stance(df),
                weight=s.weight,
                rationale=s.rationale(df),
            )
            for s in strategies
        ]
        if cfg.rl.enabled:
            if rl is not None:
                try:
                    stance, rationale = rl.stance(asset.symbol, df)
                    voices.append(Voice("rl_ppo", stance, cfg.rl.weight, rationale))
                except Exception as exc:
                    print(f"warning: RL voice failed for {asset.symbol}: {exc}",
                          file=sys.stderr)
                    voices.append(Voice("rl_ppo", 0, cfg.rl.weight,
                                        f"unavailable: {exc}", available=False))
            else:
                voices.append(Voice("rl_ppo", 0, cfg.rl.weight,
                                    f"unavailable: {rl_error}", available=False))
        out[asset.symbol] = voices
    return out
