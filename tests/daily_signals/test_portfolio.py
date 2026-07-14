"""Paper portfolio: bootstrap, transitions, fees, cash safety, benchmark."""

import pandas as pd
import pytest

from daily_signals.ensemble import Voice, combine_voices
from daily_signals.portfolio import buy_hold_nav, update_paper_portfolio
from daily_signals.state import StateStore


def price_frame(closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({
        "date": idx, "open": closes, "high": closes, "low": closes,
        "close": [float(c) for c in closes], "volume": 1000.0,
    })


def result_for(stance, prev=0):
    return combine_voices(
        [Voice("trend_sma", stance, 1.0, "r"), Voice("momentum_roc", stance, 1.0, "r")],
        prev_stance=prev,
    )


@pytest.fixture
def two_asset_config(tmp_path):
    from daily_signals.config import (
        AssetConfig, BacktestConfig, Config, DataConfig, EmailConfig,
        EnsembleConfig, PortfolioConfig, ReportConfig, RLConfig, StrategySpec,
    )

    return Config(
        base_currency="USD",
        assets=(AssetConfig("BTC-USD", 0.6), AssetConfig("ETH-USD", 0.3)),
        data=DataConfig(),
        strategies=(StrategySpec(name="trend_sma"),),
        rl=RLConfig(),
        ensemble=EnsembleConfig(),
        backtest=BacktestConfig(),
        portfolio=PortfolioConfig(initial_cash=10_000, commission_bps=100),
        report=ReportConfig(),
        email=EmailConfig(),
        state_dir=str(tmp_path / "state"),
    )


def test_first_run_bootstraps_deposit_and_buys(two_asset_config):
    cfg = two_asset_config
    store = StateStore(cfg.state_dir)
    data = {"BTC-USD": price_frame([100, 100]), "ETH-USD": price_frame([10, 10])}
    results = {"BTC-USD": result_for(1), "ETH-USD": result_for(0)}

    trades = update_paper_portfolio(cfg, store, data["BTC-USD"]["date"].iloc[-1],
                                    results, data)

    assert [t["side"] for t in trades] == ["BUY"]
    ledger = store.read_ledger()
    assert ledger["side"].tolist() == ["DEPOSIT", "BUY"]

    # buy: 60% of 10k nav at 1% fee, fill at close 100
    buy = trades[0]
    assert buy["qty"] * buy["fill_price"] == pytest.approx(6_000)
    assert buy["fee"] == pytest.approx(60.0)

    cash, positions = store.holdings()
    assert cash == pytest.approx(10_000 - 6_000 - 60)
    assert positions["BTC-USD"] == pytest.approx(60.0)

    equity = store.read_equity()
    assert len(equity) == 1
    assert equity["nav"].iloc[0] == pytest.approx(10_000 - 60)  # fee is the only loss
    assert equity["buy_hold_nav"].iloc[0] == pytest.approx(10_000)


def test_flip_to_flat_sells_everything(two_asset_config):
    cfg = two_asset_config
    store = StateStore(cfg.state_dir)
    data = {"BTC-USD": price_frame([100, 100, 110]),
            "ETH-USD": price_frame([10, 10, 10])}
    d1, d2 = data["BTC-USD"]["date"].iloc[-2], data["BTC-USD"]["date"].iloc[-1]

    update_paper_portfolio(cfg, store, d1,
                           {"BTC-USD": result_for(1), "ETH-USD": result_for(0)}, data)
    trades = update_paper_portfolio(
        cfg, store, d2,
        {"BTC-USD": result_for(0, prev=1), "ETH-USD": result_for(0)}, data)

    assert [t["side"] for t in trades] == ["SELL"]
    sell = trades[0]
    assert sell["fill_price"] == pytest.approx(110)
    cash, positions = store.holdings()
    assert positions == {}
    # bought ~59.4 BTC-equivalent units at 100 (6000/101 spend model): recompute
    ledger = store.read_ledger()
    buy_qty = float(ledger.loc[ledger["side"] == "BUY", "qty"].iloc[0])
    expected_cash = (10_000 - buy_qty * 100 * 1.01) + buy_qty * 110 * 0.99
    assert cash == pytest.approx(expected_cash)


def test_no_stance_change_books_no_trades_but_marks_equity(two_asset_config):
    cfg = two_asset_config
    store = StateStore(cfg.state_dir)
    data = {"BTC-USD": price_frame([100, 100, 120]),
            "ETH-USD": price_frame([10, 10, 10])}
    d1, d2 = data["BTC-USD"]["date"].iloc[-2], data["BTC-USD"]["date"].iloc[-1]
    results = {"BTC-USD": result_for(1), "ETH-USD": result_for(0)}

    update_paper_portfolio(cfg, store, d1, results, data)
    trades = update_paper_portfolio(cfg, store, d2, results, data)

    assert trades == []
    assert len(store.read_ledger()) == 2  # deposit + first buy only
    equity = store.read_equity()
    assert len(equity) == 2
    # BTC +20%: nav rose by 0.2 * invested value
    assert equity["nav"].iloc[-1] > equity["nav"].iloc[0]


def test_cash_never_goes_negative_when_fully_allocated(tmp_path):
    from daily_signals.config import (
        AssetConfig, BacktestConfig, Config, DataConfig, EmailConfig,
        EnsembleConfig, PortfolioConfig, ReportConfig, RLConfig, StrategySpec,
    )

    cfg = Config(
        base_currency="USD",
        assets=(AssetConfig("A-USD", 0.5), AssetConfig("B-USD", 0.5)),
        data=DataConfig(),
        strategies=(StrategySpec(name="trend_sma"),),
        rl=RLConfig(),
        ensemble=EnsembleConfig(),
        backtest=BacktestConfig(),
        portfolio=PortfolioConfig(initial_cash=10_000, commission_bps=100),
        report=ReportConfig(),
        email=EmailConfig(),
        state_dir=str(tmp_path / "state"),
    )
    store = StateStore(cfg.state_dir)
    data = {"A-USD": price_frame([100, 100]), "B-USD": price_frame([10, 10])}
    results = {"A-USD": result_for(1), "B-USD": result_for(1)}

    update_paper_portfolio(cfg, store, data["A-USD"]["date"].iloc[-1], results, data)

    cash, positions = store.holdings()
    assert cash >= 0
    assert set(positions) == {"A-USD", "B-USD"}
    # the second buy was scaled down to what cash (net of fees) allowed
    equity = store.read_equity()
    assert equity["cash"].iloc[0] >= 0


def test_buy_hold_benchmark_tracks_inception_weights(two_asset_config):
    cfg = two_asset_config
    store = StateStore(cfg.state_dir)
    data = {"BTC-USD": price_frame([100, 100, 150]),
            "ETH-USD": price_frame([10, 10, 5])}
    d1, d2 = data["BTC-USD"]["date"].iloc[-2], data["BTC-USD"]["date"].iloc[-1]
    results = {"BTC-USD": result_for(1), "ETH-USD": result_for(0)}

    update_paper_portfolio(cfg, store, d1, results, data)
    bh = buy_hold_nav(cfg, store, data, d2)
    # 60% BTC (+50%), 30% ETH (-50%), 10% cash
    assert bh == pytest.approx(10_000 * (0.6 * 1.5 + 0.3 * 0.5 + 0.1))
