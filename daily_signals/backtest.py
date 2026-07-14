"""Vectorized daily backtester and the multi-asset tearsheet.

Execution model (no lookahead): the stance decided on close(t) becomes the
position during bar t+1 (``pos = stance.shift(1)``) and earns the
close(t)->close(t+1) return, i.e. fills happen at the decision close — in a
24/7 crypto market the price seconds after the UTC close is the close.
Costs: ``commission_bps`` per side on every position change, covering fees
plus slippage. This is deliberately the same convention the live paper
ledger uses, so backtest and track record are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from daily_signals.ensemble import ensemble_series
from daily_signals.metrics import summarize
from daily_signals.strategies import build_strategies

ENSEMBLE = "ensemble"
BUY_HOLD = "buy_hold"
PORTFOLIO = "PORTFOLIO"


@dataclass
class BacktestResult:
    equity: pd.Series  # NAV, starts at initial_cash
    returns: pd.Series  # net daily strategy returns
    positions: pd.Series  # executed position (after the 1-bar lag)
    num_trades: int


def run_backtest(prices: pd.Series, stances: pd.Series, commission_bps: float,
                 start: Optional[str] = None,
                 initial_cash: float = 1.0) -> BacktestResult:
    """Backtest a 0/1 stance series against a close-price series.

    Both series must share an index (dates). Indicators should be computed on
    full history *before* slicing: ``start`` only trims the evaluation window.
    """
    stances = stances.reindex(prices.index).fillna(0).astype(int)
    positions = stances.shift(1).fillna(0).astype(int)
    market_returns = prices.pct_change().fillna(0.0)
    cost_rate = commission_bps / 10_000.0
    costs = positions.diff().abs().fillna(positions.abs()) * cost_rate
    net_returns = positions * market_returns - costs

    if start is not None:
        window = net_returns.index >= pd.Timestamp(start)
        net_returns = net_returns[window]
        positions = positions[window]

    equity = initial_cash * (1 + net_returns).cumprod()
    num_trades = int(positions.diff().abs().fillna(positions.abs()).sum())
    return BacktestResult(
        equity=equity, returns=net_returns, positions=positions,
        num_trades=num_trades,
    )


def voice_stance_frame(cfg, df: pd.DataFrame,
                       extra: Optional[Dict[str, pd.Series]] = None):
    """All voice stance histories for one asset, plus their vote weights."""
    indexed = df.set_index("date")
    series: Dict[str, pd.Series] = {}
    weights: Dict[str, float] = {}
    for strategy in build_strategies(cfg):
        stances = strategy.stance_series(df)
        stances.index = indexed.index
        series[strategy.name] = stances
        weights[strategy.name] = strategy.weight
    if extra:
        for name, stances in extra.items():
            # keep NaN outside the voice's coverage: absent from the vote there
            series[name] = stances.reindex(indexed.index).astype(float)
            weights[name] = cfg.rl.weight
    return indexed, series, weights


def tearsheet(cfg, data: Dict[str, pd.DataFrame],
              extra_series: Optional[Dict[str, Dict[str, pd.Series]]] = None
              ) -> pd.DataFrame:
    """Per-asset, per-voice performance table, plus ensemble, buy-and-hold
    and a blended portfolio row. ``extra_series`` can inject the RL voice's
    replayed stance history: {symbol: {"rl_ppo": Series}}.
    """
    bps = cfg.backtest.commission_bps
    start = cfg.backtest.start
    rows = []
    portfolio_returns: Optional[pd.Series] = None
    portfolio_bench: Optional[pd.Series] = None

    for asset in cfg.assets:
        df = data[asset.symbol]
        indexed, series, weights = voice_stance_frame(
            cfg, df, (extra_series or {}).get(asset.symbol)
        )
        prices = indexed["close"]

        always_long = pd.Series(1, index=prices.index)
        bench = run_backtest(prices, always_long, bps, start=start)

        series[ENSEMBLE] = ensemble_series(
            series, weights, long_threshold=cfg.ensemble.long_threshold
        )
        for voice_name, stances in series.items():
            result = run_backtest(prices, stances, bps, start=start)
            row = summarize(result.returns, result.positions,
                            benchmark_returns=bench.returns)
            rows.append({"asset": asset.symbol, "voice": voice_name, **row})
            if voice_name == ENSEMBLE:
                weighted = asset.weight * result.returns
                portfolio_returns = (
                    weighted if portfolio_returns is None
                    else portfolio_returns.add(weighted, fill_value=0.0)
                )

        bench_row = summarize(bench.returns, bench.positions,
                              benchmark_returns=bench.returns)
        rows.append({"asset": asset.symbol, "voice": BUY_HOLD, **bench_row})
        weighted_bench = asset.weight * bench.returns
        portfolio_bench = (
            weighted_bench if portfolio_bench is None
            else portfolio_bench.add(weighted_bench, fill_value=0.0)
        )

    if portfolio_returns is not None:
        ones = pd.Series(1, index=portfolio_returns.index)
        for voice_name, rets in ((ENSEMBLE, portfolio_returns),
                                 (BUY_HOLD, portfolio_bench)):
            row = summarize(rets, ones, benchmark_returns=portfolio_bench)
            # per-asset trade stats don't apply to the blended portfolio
            for key in ("num_trades", "trades_per_month", "win_rate", "exposure"):
                row[key] = float("nan")
            rows.append({"asset": PORTFOLIO, "voice": voice_name, **row})
    return pd.DataFrame(rows)


def equity_curves(cfg, data: Dict[str, pd.DataFrame],
                  extra_series: Optional[Dict[str, Dict[str, pd.Series]]] = None
                  ) -> pd.DataFrame:
    """Ensemble vs buy-and-hold equity per asset plus the blended portfolio,
    normalized to 1.0 — used by the report charts."""
    bps = cfg.backtest.commission_bps
    start = cfg.backtest.start
    curves: Dict[str, pd.Series] = {}
    blended: Optional[pd.Series] = None
    blended_bench: Optional[pd.Series] = None
    for asset in cfg.assets:
        df = data[asset.symbol]
        indexed, series, weights = voice_stance_frame(
            cfg, df, (extra_series or {}).get(asset.symbol)
        )
        prices = indexed["close"]
        ens = ensemble_series(series, weights,
                              long_threshold=cfg.ensemble.long_threshold)
        result = run_backtest(prices, ens, bps, start=start)
        bench = run_backtest(prices, pd.Series(1, index=prices.index), bps,
                             start=start)
        curves[f"{asset.symbol} ensemble"] = result.equity
        curves[f"{asset.symbol} buy&hold"] = bench.equity
        weighted = asset.weight * result.returns
        weighted_bench = asset.weight * bench.returns
        blended = weighted if blended is None else blended.add(weighted, fill_value=0)
        blended_bench = (
            weighted_bench if blended_bench is None
            else blended_bench.add(weighted_bench, fill_value=0)
        )
    if blended is not None:
        curves[f"{PORTFOLIO} ensemble"] = (1 + blended).cumprod()
        curves[f"{PORTFOLIO} buy&hold"] = (1 + blended_bench).cumprod()
    return pd.DataFrame(curves)


_PERCENT_COLUMNS = ["total_return", "cagr", "max_drawdown", "win_rate",
                    "exposure", "vs_buy_hold"]


def render_tearsheet_text(sheet: pd.DataFrame) -> str:
    """Human-readable tearsheet for the CLI."""
    if sheet.empty:
        return "(no results)"
    out = sheet.copy()
    for col in _PERCENT_COLUMNS:
        if col in out:
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%"
            )
    for col in ("sharpe", "sortino", "trades_per_month"):
        if col in out:
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.2f}"
            )
    if "num_trades" in out:
        out["num_trades"] = out["num_trades"].map(
            lambda v: "" if pd.isna(v) else f"{int(v)}"
        )
    return out.to_string(index=False)
