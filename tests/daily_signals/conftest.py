"""Shared fixtures for daily_signals tests.

Everything here runs offline: configs point at the committed fixture CSVs
(``data.source: fixture``) and a socket guard makes any accidental network
call fail fast.
"""

import socket
from pathlib import Path

import pytest

from daily_signals.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def no_network(monkeypatch):
    """Make any socket connection attempt raise instead of touching the net."""

    def guard(*args, **kwargs):
        raise AssertionError("network access attempted during an offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    return guard


@pytest.fixture
def fixture_config(tmp_path):
    """A full Config pointing at committed fixtures, writing under tmp_path."""
    config_text = f"""
base_currency: USD
assets:
  - {{symbol: BTC-USD, weight: 0.5}}
  - {{symbol: ETH-USD, weight: 0.3}}
  - {{symbol: SOL-USD, weight: 0.2}}
data:
  source: fixture
  fixture_dir: {FIXTURE_DIR}
  cache_dir: {tmp_path / 'cache'}
strategies:
  trend_sma:    {{enabled: true, fast: 20, slow: 100, weight: 1.0}}
  momentum_roc: {{enabled: true, lookback: 90, weight: 1.0}}
  meanrev_rsi:  {{enabled: true, period: 14, enter_below: 30, exit_above: 55, weight: 0.5}}
rl:
  enabled: false
backtest:
  start: "2018-06-01"
  commission_bps: 15
  initial_cash: 100000
portfolio:
  initial_cash: 100000
  commission_bps: 15
report:
  site_dir: {tmp_path / 'site'}
  outbox_dir: {tmp_path / 'outbox'}
  title: Test Signals
email:
  enabled: false
state_dir: {tmp_path / 'state'}
"""
    path = tmp_path / "signals.yaml"
    path.write_text(config_text)
    return load_config(path)


@pytest.fixture
def fixture_config_path(fixture_config, tmp_path):
    return tmp_path / "signals.yaml"
