"""Aggregation helpers for turning lemma-level measurements into language summaries."""

from .hierarchical import HierarchicalAggregator
from .pooling import aggregate_language_scores
from .records import LanguageSummary, LemmaMetricRecord
from .builders import PriorConfig

__all__ = [
    "HierarchicalAggregator",
    "PriorConfig",
    "aggregate_language_scores",
    "LanguageSummary",
    "LemmaMetricRecord",
]
