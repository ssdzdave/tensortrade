"""Ensemble vote logic: weighting, sticky ties, unavailable voices."""

import pandas as pd
import pytest

from daily_signals.ensemble import (
    Voice,
    combine_voices,
    compute_voices,
    ensemble_series,
)


def voice(name, stance, weight=1.0, available=True):
    return Voice(name=name, stance=stance, weight=weight,
                 rationale=f"{name} says {stance}", available=available)


class TestCombineVoices:
    def test_unanimous_long(self):
        result = combine_voices([voice("a", 1), voice("b", 1)], prev_stance=0)
        assert result.stance == 1 and result.changed
        assert result.score == 1.0
        assert result.n_long == 2 and result.n_voices == 2

    def test_majority_by_weight(self):
        # 1.0 long vs 0.5 flat -> score 2/3 -> LONG
        result = combine_voices([voice("a", 1, 1.0), voice("b", 0, 0.5)])
        assert result.stance == 1
        assert result.score == pytest.approx(2 / 3)

    def test_exact_tie_keeps_previous_stance(self):
        voices = [voice("a", 1), voice("b", 0)]
        assert combine_voices(voices, prev_stance=1).stance == 1
        assert combine_voices(voices, prev_stance=0).stance == 0
        assert not combine_voices(voices, prev_stance=1).changed
        assert "tie" in combine_voices(voices, prev_stance=1).vote_summary()

    def test_unavailable_voice_excluded_from_denominator(self):
        voices = [voice("a", 1), voice("rl", 0, weight=5.0, available=False)]
        result = combine_voices(voices, prev_stance=0)
        assert result.stance == 1  # the huge unavailable weight is ignored
        assert result.n_voices == 1

    def test_all_unavailable_keeps_previous(self):
        voices = [voice("rl", 1, available=False)]
        assert combine_voices(voices, prev_stance=1).stance == 1
        assert combine_voices(voices, prev_stance=0).stance == 0

    def test_below_threshold_is_flat(self):
        result = combine_voices([voice("a", 0), voice("b", 0), voice("c", 1)],
                                prev_stance=1)
        assert result.stance == 0 and result.changed


class TestEnsembleSeries:
    def test_matches_sequential_combine(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        a = pd.Series([1, 1, 0, 0, 1, 1], index=idx)
        b = pd.Series([1, 0, 0, 1, 1, 0], index=idx)
        weights = {"a": 1.0, "b": 1.0}

        series = ensemble_series({"a": a, "b": b}, weights)

        prev = 0
        expected = []
        for i in range(len(idx)):
            result = combine_voices(
                [voice("a", int(a.iloc[i])), voice("b", int(b.iloc[i]))],
                prev_stance=prev,
            )
            prev = result.stance
            expected.append(result.stance)
        assert series.tolist() == expected
        # ties at bars 1 and 3 stick to the previous stance (1 after bar 0)
        assert series.tolist() == [1, 1, 0, 0, 1, 1]

    def test_weighted_series(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        a = pd.Series([1, 1], index=idx)
        b = pd.Series([0, 0], index=idx)
        heavy_b = ensemble_series({"a": a, "b": b}, {"a": 1.0, "b": 3.0})
        assert heavy_b.tolist() == [0, 0]

    def test_requires_voices_and_positive_weight(self):
        with pytest.raises(ValueError):
            ensemble_series({}, {})
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        with pytest.raises(ValueError):
            ensemble_series({"a": pd.Series([1, 0], index=idx)}, {"a": 0.0})


def test_compute_voices_on_fixtures(fixture_config, no_network):
    from daily_signals.data.cache import load_all

    data = load_all(fixture_config)
    voices_by_asset = compute_voices(fixture_config, data)

    assert set(voices_by_asset) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    for asset, voices in voices_by_asset.items():
        names = [v.name for v in voices]
        assert names == ["trend_sma", "momentum_roc", "meanrev_rsi"]  # rl disabled
        for v in voices:
            assert v.stance in (0, 1)
            assert v.rationale


def test_compute_voices_rl_enabled_but_missing_degrades(fixture_config, no_network):
    from dataclasses import replace

    from daily_signals.data.cache import load_all

    cfg = replace(fixture_config, rl=replace(fixture_config.rl, enabled=True))
    data = load_all(cfg)
    voices_by_asset = compute_voices(cfg, data)

    for voices in voices_by_asset.values():
        rl = [v for v in voices if v.name == "rl_ppo"]
        assert len(rl) == 1
        assert not rl[0].available
        assert "unavailable" in rl[0].rationale
        # the ensemble still works without it
        result = combine_voices(voices, prev_stance=0)
        assert result.stance in (0, 1)
