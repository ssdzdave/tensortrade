"""Train the per-asset PPO voice on daily bars from the signals cache.

Adapted from examples/training/train_best.py (its Optuna-tuned
hyperparameters, PBR reward, commission-during-training discipline) with
three product-grade changes: daily bars from the same cache the signals
use, checkpoints persisted under models/<ASSET>/ (never /tmp, never
deleted), and a manifest.json describing exactly what was trained so
inference can rebuild the same env.

Usage:
    python -m daily_signals.rl.train_daily --asset BTC-USD [--config ...]
                                           [--iterations 60] [--all]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from daily_signals.config import Config, load_config
from daily_signals.rl.envs import create_frame_env
from daily_signals.rl.features import featurize

ENV_NAME = "DailySignalsEnv"
VAL_DAYS = 60
TEST_DAYS = 60
EVAL_EVERY = 10
EVAL_EPISODES = 5

# Optuna-tuned hyperparameters from examples/training/train_optuna.py.
# NOTE: ray 2.37's old API stack spells these sgd_minibatch_size/num_sgd_iter
# (the minibatch_size/num_epochs spellings in some example scripts are for
# newer ray and fail here).
BEST_HPS = dict(
    lr=3.29e-05,
    gamma=0.992,
    lambda_=0.9,
    clip_param=0.123,
    entropy_coeff=0.015,
    train_batch_size=2000,
    sgd_minibatch_size=256,
    num_sgd_iter=7,
    vf_clip_param=100.0,
)


def evaluate(algo, frame: pd.DataFrame, feature_cols: List[str],
             window_size: int, commission: float, episodes: int = EVAL_EPISODES,
             initial_cash: float = 10_000) -> float:
    """Average episode P&L of the current policy on ``frame``."""
    pnls = []
    for _ in range(episodes):
        env = create_frame_env({
            "frame": frame,
            "feature_cols": feature_cols,
            "window_size": window_size,
            "commission": commission,
            "initial_cash": initial_cash,
            "max_allowed_loss": 0.9,
        })
        obs, _ = env.reset()
        done = truncated = False
        while not done and not truncated:
            action = algo.compute_single_action(obs)
            obs, _, done, truncated, _ = env.step(action)
        pnls.append(env.portfolio.net_worth - initial_cash)
    return float(np.mean(pnls))


def train_asset(cfg: Config, symbol: str, iterations: int = 60,
                window_size: int = 10, commission: float = 0.003) -> Path:
    """Train one asset's PPO voice; returns the model directory."""
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from daily_signals.data.cache import load
    from daily_signals.state import git_sha

    raw = load(cfg, symbol)
    frame, feature_cols = featurize(raw)
    if len(frame) < VAL_DAYS + TEST_DAYS + 200:
        raise SystemExit(
            f"{symbol}: only {len(frame)} usable bars after warmup; need at "
            f"least {VAL_DAYS + TEST_DAYS + 200}. Fetch more history first."
        )

    test = frame.iloc[-TEST_DAYS:].reset_index(drop=True)
    val = frame.iloc[-(TEST_DAYS + VAL_DAYS):-TEST_DAYS].reset_index(drop=True)
    train = frame.iloc[:-(TEST_DAYS + VAL_DAYS)].reset_index(drop=True)
    print(f"{symbol}: train={len(train)} val={len(val)} test={len(test)} daily bars")

    env_config = {
        "frame": train,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "commission": commission,  # training WITH commission teaches discipline
        "initial_cash": 10_000,
        "max_allowed_loss": 0.4,
        "symbol": symbol,
    }

    ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    register_env(ENV_NAME, create_frame_env)

    ppo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=0)  # tiny dataset: sample on the driver
        .training(model={"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"},
                  **BEST_HPS)
        .resources(num_gpus=0)
    )
    algo = ppo_config.build()

    model_dir = Path(cfg.rl.models_dir) / symbol
    checkpoint_dir = model_dir / "ppo"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("-inf")
    best_iteration = 0
    for i in range(iterations):
        algo.train()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == iterations:
            val_pnl = evaluate(algo, val, feature_cols, window_size, commission)
            marker = ""
            if val_pnl > best_val:
                best_val = val_pnl
                best_iteration = i + 1
                algo.save(str(checkpoint_dir.resolve()))
                marker = " * saved"
            print(f"  iter {i + 1:3d}: val P&L ${val_pnl:+,.0f} "
                  f"(best ${best_val:+,.0f} @ {best_iteration}){marker}")

    algo.restore(str(checkpoint_dir.resolve()))
    test_pnl = evaluate(algo, test, feature_cols, window_size, commission=0.001,
                        episodes=10)
    test_bh = 10_000 * (test["close"].iloc[-1] / test["close"].iloc[0] - 1)
    print(f"{symbol}: held-out test P&L ${test_pnl:+,.0f} "
          f"(buy & hold ${test_bh:+,.0f}) at 10 bps")

    manifest = {
        "symbol": symbol,
        "checkpoint_relpath": "ppo",
        "policy_relpath": "ppo/policies/default_policy",
        "feature_cols": feature_cols,
        "window_size": window_size,
        "train_commission": commission,
        "data_start": str(frame["date"].iloc[0].date()),
        "data_end": str(frame["date"].iloc[-1].date()),
        "val_pnl": round(best_val, 2),
        "test_pnl_10bps": round(test_pnl, 2),
        "test_buy_hold": round(test_bh, 2),
        "best_iteration": best_iteration,
        "iterations": iterations,
        "trained_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{symbol}: checkpoint + manifest written to {model_dir}")

    algo.stop()
    return model_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/signals.yaml")
    parser.add_argument("--asset", help="e.g. BTC-USD")
    parser.add_argument("--all", action="store_true",
                        help="train every configured asset")
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.all:
        symbols = [a.symbol for a in cfg.assets]
    elif args.asset:
        symbols = [args.asset]
    else:
        parser.error("pass --asset BTC-USD or --all")

    for symbol in symbols:
        train_asset(cfg, symbol, iterations=args.iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
