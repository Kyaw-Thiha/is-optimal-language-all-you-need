"""Shared data records for metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LemmaMetricRecord:
    """Single lemma-level metric observation."""

    language: str
    lemma: str
    metric: str
    value: float
    variance: Optional[float] = None
    support_size: Optional[int] = None


@dataclass(frozen=True)
class LanguageSummary:
    """Posterior summary for a language-level aggregation."""

    language: str
    mean: float
    lower: float
    upper: float
