"""Unit tests for DDI metrics and threshold policies."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.ddi import DDIConfig, compute_ddi
from src.metrics.ddi_policy import FixedThresholdPolicy, PercentileThresholdPolicy, ThresholdPolicy


# ---------------------------------------------------------------------------
# Threshold policy tests


def test_fixed_threshold_returns_constant_value() -> None:
    policy = FixedThresholdPolicy(0.75)
    scores = {0: 0.2, 1: 0.6}
    assert policy.derive(scores) == pytest.approx(0.75)


def test_fixed_threshold_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        FixedThresholdPolicy(float("nan"))
    with pytest.raises(ValueError):
        FixedThresholdPolicy(float("inf"))


def test_fixed_threshold_requires_scores() -> None:
    policy = FixedThresholdPolicy(0.5)
    with pytest.raises(ValueError):
        policy.derive({})


def test_percentile_threshold_computes_percentile() -> None:
    scores = {0: 0.2, 1: 0.6, 2: 0.9}
    policy = PercentileThresholdPolicy(60.0)
    expected = np.percentile(list(scores.values()), 60.0)
    assert policy.derive(scores) == pytest.approx(expected)


def test_percentile_threshold_validates_range() -> None:
    with pytest.raises(ValueError):
        PercentileThresholdPolicy(-1.0)
    with pytest.raises(ValueError):
        PercentileThresholdPolicy(101.0)


def test_percentile_threshold_handles_bad_inputs() -> None:
    policy = PercentileThresholdPolicy(50.0)
    with pytest.raises(ValueError):
        policy.derive({})
    with pytest.raises(ValueError):
        policy.derive({0: float("nan")})


# ---------------------------------------------------------------------------
# DDI computation tests


class DummyPolicy:
    """Simple ThresholdPolicy implementation for testing."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.seen_scores: dict[int, float] | None = None

    def derive(self, scores: Mapping[int, float]) -> float:
        self.seen_scores = dict(scores)
        return self.value


def test_compute_ddi_returns_first_layer_meeting_threshold() -> None:
    scores = {0: 0.25, 1: 0.55, 2: 0.78}
    config = DDIConfig(threshold_policy=FixedThresholdPolicy(0.6))
    result = compute_ddi(scores, config)
    assert result.layer == 2
    assert result.found is True
    assert result.threshold == pytest.approx(0.6)
    assert list(result.scores.keys()) == [0, 1, 2]


def test_compute_ddi_respects_layer_bounds() -> None:
    scores = {-1: 0.4, 0: 0.5, 1: 0.7}
    config = DDIConfig(min_layer=0, max_layer=1, threshold_policy=FixedThresholdPolicy(0.6))
    result = compute_ddi(scores, config)
    assert result.layer == 1
    assert all(layer >= 0 for layer in result.scores)
    assert all(layer <= 1 for layer in result.scores)


def test_compute_ddi_returns_none_when_threshold_not_met() -> None:
    scores = {0: 0.1, 1: 0.2, 2: 0.3}
    config = DDIConfig(threshold_policy=FixedThresholdPolicy(0.5))
    result = compute_ddi(scores, config)
    assert result.layer is None
    assert result.found is False


def test_compute_ddi_raises_on_empty_scores() -> None:
    with pytest.raises(ValueError):
        compute_ddi({})


def test_compute_ddi_raises_when_filtered_empty() -> None:
    scores = {0: 0.2}
    config = DDIConfig(min_layer=1, max_layer=1)
    with pytest.raises(ValueError):
        compute_ddi(scores, config)


def test_compute_ddi_detects_non_finite_values() -> None:
    scores = {0: 0.2, 1: float("nan")}
    with pytest.raises(ValueError):
        compute_ddi(scores)


def test_compute_ddi_uses_custom_policy() -> None:
    scores = {0: 0.4, 1: 0.45, 2: 0.5}
    policy = DummyPolicy(0.46)
    config = DDIConfig(threshold_policy=policy)
    result = compute_ddi(scores, config)
    assert result.layer == 2
    assert policy.seen_scores == result.scores
