"""Threshold policies used when computing the Disambiguation Depth Index (DDI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class ThresholdPolicy(Protocol):
    """Strategy object that produces a reliability threshold from layer scores."""

    def derive(self, scores: Mapping[int, float]) -> float:
        """Return the threshold τ to compare against S(L) values."""
        return 0.0


@dataclass(frozen=True)
class FixedThresholdPolicy:
    """Always returns the same threshold value τ."""

    threshold: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("Fixed threshold must be a finite number.")

    def derive(self, scores: Mapping[int, float]) -> float:
        if not scores:
            raise ValueError("Cannot derive a fixed threshold without any scores.")
        return float(self.threshold)


@dataclass(frozen=True)
class PercentileThresholdPolicy:
    """Sets τ to the specified percentile of the provided score distribution."""

    percentile: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError("Percentile must fall within [0, 100].")

    def derive(self, scores: Mapping[int, float]) -> float:
        if not scores:
            raise ValueError("Cannot derive percentile threshold without any scores.")
        values = np.asarray(list(scores.values()), dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("Percentile threshold cannot be computed with NaN/inf scores.")
        return float(np.percentile(values, self.percentile))
