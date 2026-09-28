"""TensorTrade env builder for the daily RL voice.

Same construction as examples/training/train_best.py (BSH actions, PBR
reward) but fed a DataFrame directly — no /tmp CSV round-trip — and
parameterized per asset. Imports tensortrade (heavy) but not ray, so the
inference path can build replay envs without an RLlib session.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def create_frame_env(config: Dict[str, Any]):
    """Build a TradingEnv from {"frame": DataFrame, "feature_cols": [...]}.

    Optional keys: window_size (10), commission (0.003), initial_cash
    (10_000), max_allowed_loss (0.4), symbol ("BTC-USD").
    """
    import tensortrade.env.default as default
    from tensortrade.env.default.actions import BSH
    from tensortrade.env.default.rewards import PBR
    from tensortrade.feed.core import DataFeed, Stream
    from tensortrade.oms.exchanges import Exchange, ExchangeOptions
    from tensortrade.oms.instruments import USD, Instrument
    from tensortrade.oms.services.execution.simulated import execute_order
    from tensortrade.oms.wallets import Portfolio, Wallet

    frame: pd.DataFrame = config["frame"]
    feature_cols = list(config["feature_cols"])
    symbol = config.get("symbol", "BTC-USD")
    base_code = symbol.split("-")[0]
    base = Instrument(base_code, 8, base_code)

    price = Stream.source(list(frame["close"]), dtype="float").rename(
        f"USD-{base_code}"
    )
    exchange = Exchange(
        "exchange",
        service=execute_order,
        options=ExchangeOptions(commission=config.get("commission", 0.003)),
    )(price)

    cash = Wallet(exchange, config.get("initial_cash", 10_000) * USD)
    asset = Wallet(exchange, 0 * base)
    portfolio = Portfolio(USD, [cash, asset])

    features = [
        Stream.source(list(frame[c]), dtype="float").rename(c)
        for c in feature_cols
    ]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = PBR(price=price)
    action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)

    env = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config.get("window_size", 10),
        max_allowed_loss=config.get("max_allowed_loss", 0.4),
    )
    env.portfolio = portfolio
    return env
