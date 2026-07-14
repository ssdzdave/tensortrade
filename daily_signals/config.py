"""Configuration loading and validation for the daily signals pipeline.

Config lives in a YAML file (see ``config/signals.example.yaml``); secrets are
never stored there — they are read from environment variables at the point of
use (e.g. the SMTP password from ``SIGNALS_SMTP_PASSWORD``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when the YAML config is missing or invalid."""


@dataclass(frozen=True)
class AssetConfig:
    symbol: str  # exchange product id, e.g. "BTC-USD"
    weight: float  # share of capital allocated when stance is LONG

    @property
    def base(self) -> str:
        return self.symbol.split("-")[0]


@dataclass(frozen=True)
class DataConfig:
    source: str = "coinbase"  # coinbase | ccxt | fixture
    ccxt_exchange: str = "kraken"
    cache_dir: str = "data/cache"
    fixture_dir: str = "tests/daily_signals/fixtures"
    min_history_days: int = 400


@dataclass(frozen=True)
class StrategySpec:
    name: str
    enabled: bool = True
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RLConfig:
    enabled: bool = False
    models_dir: str = "models"
    weight: float = 1.0


@dataclass(frozen=True)
class EnsembleConfig:
    # score > long_threshold => LONG, score < long_threshold => FLAT,
    # score == long_threshold => keep yesterday's stance.
    long_threshold: float = 0.5


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2019-01-01"
    commission_bps: float = 15.0  # per side, in basis points
    initial_cash: float = 100_000.0


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 15.0


@dataclass(frozen=True)
class ReportConfig:
    site_dir: str = "reports/site"
    outbox_dir: str = "reports/outbox"
    title: str = "Daily Crypto Signals"
    site_url: str = ""  # public URL of the published site, if any


@dataclass(frozen=True)
class EmailConfig:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    from_addr: str = ""
    username: str = ""  # defaults to from_addr when empty
    bcc: tuple = ()
    subject_prefix: str = "[Daily Signals]"
    password_env: str = "SIGNALS_SMTP_PASSWORD"

    def password(self) -> str:
        value = os.environ.get(self.password_env, "")
        if not value:
            raise ConfigError(
                f"SMTP password not found: set the {self.password_env} "
                "environment variable (never put it in the YAML config)."
            )
        return value


@dataclass(frozen=True)
class Config:
    base_currency: str
    assets: tuple
    data: DataConfig
    strategies: tuple
    rl: RLConfig
    ensemble: EnsembleConfig
    backtest: BacktestConfig
    portfolio: PortfolioConfig
    report: ReportConfig
    email: EmailConfig
    state_dir: str = "state"

    def enabled_strategies(self) -> tuple:
        return tuple(s for s in self.strategies if s.enabled)


# Known strategies and their default parameters. The registry in
# daily_signals.strategies builds concrete instances from these specs.
DEFAULT_STRATEGY_PARAMS: dict = {
    "trend_sma": {"fast": 20, "slow": 100},
    "momentum_roc": {"lookback": 90},
    "meanrev_rsi": {"period": 14, "enter_below": 30, "exit_above": 55},
}


def _require(mapping: Mapping, key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return mapping[key]


def _parse_assets(raw: Any) -> tuple:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("'assets' must be a non-empty list")
    assets = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Each asset must be a mapping, got: {entry!r}")
        symbol = str(_require(entry, "symbol", "asset entry"))
        if "-" not in symbol:
            raise ConfigError(
                f"Asset symbol '{symbol}' must look like 'BTC-USD' (base-quote)"
            )
        weight = float(_require(entry, "weight", f"asset {symbol}"))
        if weight <= 0:
            raise ConfigError(f"Asset {symbol} weight must be > 0, got {weight}")
        assets.append(AssetConfig(symbol=symbol, weight=weight))
    total = sum(a.weight for a in assets)
    if total > 1.0 + 1e-9:
        raise ConfigError(
            f"Asset weights sum to {total:.4f}; must be <= 1.0 "
            "(the remainder stays in cash)"
        )
    seen = set()
    for a in assets:
        if a.symbol in seen:
            raise ConfigError(f"Duplicate asset symbol: {a.symbol}")
        seen.add(a.symbol)
    return tuple(assets)


def _parse_strategies(raw: Any) -> tuple:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("'strategies' must be a mapping of name -> options")
    specs = []
    for name, options in raw.items():
        if name not in DEFAULT_STRATEGY_PARAMS:
            known = ", ".join(sorted(DEFAULT_STRATEGY_PARAMS))
            raise ConfigError(f"Unknown strategy '{name}'. Known: {known}")
        options = dict(options or {})
        enabled = bool(options.pop("enabled", True))
        weight = float(options.pop("weight", 1.0))
        if weight < 0:
            raise ConfigError(f"Strategy {name} weight must be >= 0")
        params = dict(DEFAULT_STRATEGY_PARAMS[name])
        for key, value in options.items():
            if key not in params:
                allowed = ", ".join(sorted(params))
                raise ConfigError(
                    f"Unknown parameter '{key}' for strategy {name}. "
                    f"Allowed: {allowed}"
                )
            params[key] = value
        specs.append(
            StrategySpec(name=name, enabled=enabled, weight=weight, params=params)
        )
    if not any(s.enabled for s in specs):
        raise ConfigError("At least one strategy must be enabled")
    return tuple(specs)


def _dataclass_from(cls, raw: Any, where: str):
    raw = dict(raw or {})
    valid = {f for f in cls.__dataclass_fields__}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in '{where}'. Allowed: {sorted(valid)}"
        )
    if "bcc" in raw and raw["bcc"] is not None:
        raw["bcc"] = tuple(str(x) for x in raw["bcc"])
    if "start" in raw:
        raw["start"] = str(raw["start"])
    return cls(**raw)


def load_config(path) -> Config:
    """Load and validate the YAML config at ``path``."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. Run 'daily-signals init' to create "
            "one from config/signals.example.yaml, or pass --config."
        )
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Top level of {path} must be a mapping")

    known_top = {
        "base_currency", "assets", "data", "strategies", "rl", "ensemble",
        "backtest", "portfolio", "report", "email", "state_dir",
    }
    unknown = set(raw) - known_top
    if unknown:
        raise ConfigError(
            f"Unknown top-level key(s) {sorted(unknown)}. Allowed: {sorted(known_top)}"
        )

    return Config(
        base_currency=str(raw.get("base_currency", "USD")),
        assets=_parse_assets(_require(raw, "assets", str(path))),
        data=_dataclass_from(DataConfig, raw.get("data"), "data"),
        strategies=_parse_strategies(raw.get("strategies")),
        rl=_dataclass_from(RLConfig, raw.get("rl"), "rl"),
        ensemble=_dataclass_from(EnsembleConfig, raw.get("ensemble"), "ensemble"),
        backtest=_dataclass_from(BacktestConfig, raw.get("backtest"), "backtest"),
        portfolio=_dataclass_from(PortfolioConfig, raw.get("portfolio"), "portfolio"),
        report=_dataclass_from(ReportConfig, raw.get("report"), "report"),
        email=_dataclass_from(EmailConfig, raw.get("email"), "email"),
        state_dir=str(raw.get("state_dir", "state")),
    )
