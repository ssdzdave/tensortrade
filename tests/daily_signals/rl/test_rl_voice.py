"""RL voice round-trip: train briefly on fixtures, restore, emit stances.

Marked rllib/slow — CI's light job skips these; run them locally with
`pytest tests/daily_signals/rl -m rllib` after `pip install -e ".[rl]"`.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("ray")
pytest.importorskip("tensortrade")

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """One-iteration training run on the BTC fixture; shared by the tests."""
    from daily_signals.config import load_config
    from daily_signals.rl import train_daily

    tmp = tmp_path_factory.mktemp("rl")
    config_text = f"""
assets:
  - {{symbol: BTC-USD, weight: 1.0}}
data:
  source: fixture
  fixture_dir: {FIXTURE_DIR}
strategies:
  trend_sma: {{enabled: true, fast: 20, slow: 100}}
rl:
  enabled: true
  models_dir: {tmp / 'models'}
backtest:
  start: "2018-06-01"
state_dir: {tmp / 'state'}
report:
  site_dir: {tmp / 'site'}
  outbox_dir: {tmp / 'outbox'}
"""
    config_path = tmp / "signals.yaml"
    config_path.write_text(config_text)
    cfg = load_config(config_path)

    model_dir = train_daily.train_asset(cfg, "BTC-USD", iterations=1)
    return cfg, config_path, model_dir


@pytest.mark.rllib
@pytest.mark.slow
def test_training_persists_checkpoint_and_manifest(trained_model):
    _, _, model_dir = trained_model
    manifest = json.loads((model_dir / "manifest.json").read_text())
    assert manifest["symbol"] == "BTC-USD"
    assert (model_dir / manifest["checkpoint_relpath"]).exists()
    assert (model_dir / manifest["policy_relpath"]).exists()
    assert manifest["feature_cols"]
    assert manifest["window_size"] >= 2


@pytest.mark.rllib
@pytest.mark.slow
def test_voice_replays_deterministic_stances(trained_model):
    from daily_signals.data.cache import load
    from daily_signals.rl.voice import RLVoice

    cfg, _, _ = trained_model
    df = load(cfg, "BTC-USD")
    voice = RLVoice(cfg)

    series = voice.stance_series("BTC-USD", df)
    assert set(series.unique()) <= {0, 1}
    assert len(series) > 100
    assert series.equals(voice.stance_series("BTC-USD", df))  # deterministic

    stance, rationale = voice.stance("BTC-USD", df)
    assert stance == int(series.iloc[-1])
    assert "PPO policy" in rationale


@pytest.mark.rllib
@pytest.mark.slow
def test_run_daily_includes_rl_voice(trained_model, capsys):
    from daily_signals import cli

    _, config_path, _ = trained_model
    assert cli.main(["--config", str(config_path), "run-daily", "--dry-run"]) == 0

    state_dir = Path(json.loads(json.dumps(str(config_path.parent / "state"))))
    signals = [
        json.loads(line)
        for line in (state_dir / "signals.jsonl").read_text().splitlines()
    ]
    rl_rows = [s for s in signals if s["voice"] == "rl_ppo"]
    assert len(rl_rows) == 1
    assert rl_rows[0]["available"] is True
    assert rl_rows[0]["stance"] in (0, 1)

    site = (config_path.parent / "site" / "index.html").read_text()
    assert "RL agent (PPO)" in site
