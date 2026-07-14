"""Strategy registry: build concrete instances from config specs."""

from __future__ import annotations

from typing import List

from daily_signals.strategies.base import Strategy
from daily_signals.strategies.meanrev import RsiMeanReversion
from daily_signals.strategies.momentum import RocMomentum
from daily_signals.strategies.trend import SmaCross

REGISTRY = {
    SmaCross.name: SmaCross,
    RocMomentum.name: RocMomentum,
    RsiMeanReversion.name: RsiMeanReversion,
}


def build_strategies(cfg) -> List[Strategy]:
    """Instantiate the enabled strategies from a loaded Config."""
    strategies = []
    for spec in cfg.enabled_strategies():
        cls = REGISTRY.get(spec.name)
        if cls is None:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"Unknown strategy '{spec.name}'. Known: {known}")
        strategies.append(cls(weight=spec.weight, **spec.params))
    return strategies
