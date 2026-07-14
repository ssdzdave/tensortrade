"""Paper portfolio: turn ensemble stances into hypothetical trades.

Fills happen at the as-of close — the same price the signal was computed on,
which in a 24/7 market is executable within minutes of the UTC close and is
exactly the convention the backtester uses, keeping the live track record
comparable to the tearsheet. Trades occur only on stance transitions
(enter to the asset's configured capital share, exit to cash); there is no
daily drift rebalancing — fewer trades, less commission drag.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

import pandas as pd

from daily_signals.state import StateStore

DUST_NAV_FRACTION = 1e-6


def close_prices(data: Dict[str, pd.DataFrame], as_of) -> Dict[str, float]:
    prices = {}
    day = pd.Timestamp(as_of)
    for symbol, df in data.items():
        match = df.loc[df["date"] == day, "close"]
        if match.empty:
            raise ValueError(f"{symbol}: no bar for {day.date()} in cache")
        prices[symbol] = float(match.iloc[-1])
    return prices


def buy_hold_nav(cfg, store: StateStore, data: Dict[str, pd.DataFrame],
                 as_of) -> float:
    """NAV of the do-nothing benchmark: the configured weights bought at
    inception (first equity mark), the remainder held as cash."""
    inception = store.inception()
    initial = cfg.portfolio.initial_cash
    if inception is None:
        return initial
    nav = (1 - sum(a.weight for a in cfg.assets)) * initial
    now = pd.Timestamp(as_of)
    for asset in cfg.assets:
        df = data[asset.symbol]
        history = df.loc[df["date"] <= inception, "close"]
        entry = float(history.iloc[-1]) if not history.empty else None
        if entry is None:  # asset newer than inception: count it as cash
            nav += asset.weight * initial
            continue
        current = df.loc[df["date"] <= now, "close"]
        nav += asset.weight * initial * float(current.iloc[-1]) / entry
    return nav


def update_paper_portfolio(cfg, store: StateStore, as_of,
                           results: Dict[str, "EnsembleResult"],
                           data: Dict[str, pd.DataFrame]) -> List[dict]:
    """Book stance transitions into the ledger and mark today's equity.

    Returns the trades booked (possibly empty). Idempotency is delegated to
    the store; callers should already have skipped days that were recorded.
    """
    day = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    prices = close_prices(data, as_of)
    fee_rate = cfg.portfolio.commission_bps / 10_000.0

    if store.read_ledger().empty:
        store.append_ledger_rows([{
            "date": day, "asset": "CASH", "side": "DEPOSIT",
            "qty": cfg.portfolio.initial_cash, "fill_price": 1.0, "fee": 0.0,
            "cash_after": cfg.portfolio.initial_cash, "note": "initial deposit",
        }])

    cash, positions = store.holdings()
    nav = cash + sum(qty * prices[s] for s, qty in positions.items() if s in prices)

    trades: List[dict] = []

    # sells first so freed cash can fund the buys
    for asset in cfg.assets:
        symbol = asset.symbol
        qty = positions.get(symbol, 0.0)
        if results[symbol].stance == 0 and qty > 0:
            proceeds = qty * prices[symbol]
            fee = proceeds * fee_rate
            cash += proceeds - fee
            positions[symbol] = 0.0
            trades.append({
                "date": day, "asset": symbol, "side": "SELL", "qty": qty,
                "fill_price": prices[symbol], "fee": round(fee, 8),
                "cash_after": round(cash, 8),
                "note": f"exit: {results[symbol].vote_summary()}",
            })

    for asset in cfg.assets:
        symbol = asset.symbol
        held_value = positions.get(symbol, 0.0) * prices[symbol]
        if results[symbol].stance == 1 and held_value <= DUST_NAV_FRACTION * nav:
            target_value = asset.weight * nav
            spend = min(target_value, cash / (1 + fee_rate))
            if spend <= DUST_NAV_FRACTION * nav:
                continue
            qty = spend / prices[symbol]
            fee = spend * fee_rate
            cash -= spend + fee
            positions[symbol] = positions.get(symbol, 0.0) + qty
            trades.append({
                "date": day, "asset": symbol, "side": "BUY", "qty": qty,
                "fill_price": prices[symbol], "fee": round(fee, 8),
                "cash_after": round(cash, 8),
                "note": f"enter: {results[symbol].vote_summary()}",
            })

    store.append_ledger_rows(trades)

    cash_final, positions_final = store.holdings()
    positions_value = sum(
        qty * prices[s] for s, qty in positions_final.items() if s in prices
    )
    store.append_equity_row({
        "date": day,
        "nav": round(cash_final + positions_value, 8),
        "cash": round(cash_final, 8),
        "positions_value": round(positions_value, 8),
        "buy_hold_nav": round(buy_hold_nav(cfg, store, data, as_of), 8),
    })
    return trades
