"""Shared tuner protocol definitions for probes."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .base import BaseProbe


class ProbeTuner(Protocol):
    """Interface describing how probe tuners should behave."""

    def tune(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Inspect training data and run tuning (no-op if already tuned)."""

    def make_probe(self) -> BaseProbe:
        """Return a newly configured probe ready to fit."""
