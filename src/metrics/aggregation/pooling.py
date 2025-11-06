"""Public entry point for language-level aggregation."""

from __future__ import annotations

from typing import Iterable, Sequence

from .hierarchical import HierarchicalAggregator, PriorConfig
from .records import LanguageSummary, LemmaMetricRecord


def aggregate_language_scores(
    records: Iterable[LemmaMetricRecord],
    priors: PriorConfig | None = None,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9,
    chains: int = 4,
    cores: int | None = None,
    credible_interval: float = 0.90,
) -> Sequence[LanguageSummary]:
    """Fit the hierarchical model and return language summaries in one call."""
    aggregator = HierarchicalAggregator(
        priors=priors,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        chains=chains,
        cores=cores,
        credible_interval=credible_interval,
    )
    aggregator.fit(records)
    return aggregator.summaries()
