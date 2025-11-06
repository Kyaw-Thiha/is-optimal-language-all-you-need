"""Tests for hierarchical aggregation utilities."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.aggregation.builders import HierarchicalDataset, PriorConfig, build_dataset, build_model
from src.metrics.aggregation.hierarchical import HierarchicalAggregator
from src.metrics.aggregation.pooling import aggregate_language_scores
from src.metrics.aggregation.records import LanguageSummary, LemmaMetricRecord


# ---------------------------------------------------------------------------
# Dataset builder tests


def _make_records(metric: str = "ddi") -> list[LemmaMetricRecord]:
    return [
        LemmaMetricRecord(language="en", lemma="bank", metric=metric, value=6.0),
        LemmaMetricRecord(language="en", lemma="cell", metric=metric, value=5.5),
        LemmaMetricRecord(language="tr", lemma="düşmek", metric=metric, value=4.2),
        LemmaMetricRecord(language="tr", lemma="yüz", metric=metric, value=5.0),
    ]


def test_build_dataset_basic() -> None:
    records = _make_records()
    dataset = build_dataset(records)

    assert isinstance(dataset, HierarchicalDataset)
    assert dataset.metric_name == "ddi"
    assert list(dataset.language_labels) == ["en", "tr"]
    assert list(dataset.lemma_labels) == ["bank", "cell", "düşmek", "yüz"]
    assert dataset.values.shape == (4,)
    assert np.allclose(dataset.values, [6.0, 5.5, 4.2, 5.0])


def test_build_dataset_requires_single_metric() -> None:
    records = _make_records()
    records.append(LemmaMetricRecord(language="en", lemma="lead", metric="hl", value=7.0))
    with pytest.raises(ValueError):
        build_dataset(records)


def test_build_dataset_empty() -> None:
    with pytest.raises(ValueError):
        build_dataset([])


def test_build_model_constructs_pymc_model() -> None:
    dataset = build_dataset(_make_records())
    priors = PriorConfig()
    model = build_model(dataset, priors)
    assert model is not None
    assert {"mu_global", "mu_language", "beta_lemma"}.issubset(model.named_vars)


# ---------------------------------------------------------------------------
# Aggregator tests


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hierarchical_aggregator_produces_language_summaries() -> None:
    records = [
        LemmaMetricRecord(language="en", lemma="bank", metric="ddi", value=6.4),
        LemmaMetricRecord(language="en", lemma="cell", metric="ddi", value=6.0),
        LemmaMetricRecord(language="en", lemma="lead", metric="ddi", value=5.8),
        LemmaMetricRecord(language="tr", lemma="düşmek", metric="ddi", value=4.1),
        LemmaMetricRecord(language="tr", lemma="yazmak", metric="ddi", value=4.8),
        LemmaMetricRecord(language="tr", lemma="yüz", metric="ddi", value=4.9),
    ]

    aggregator = HierarchicalAggregator(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        credible_interval=0.90,
        random_seed=123,
    )
    aggregator.fit(records)
    summaries = aggregator.summaries()

    languages = [summary.language for summary in summaries]
    assert languages == ["en", "tr"]  # alphabetical order from builder
    assert all(isinstance(summary, LanguageSummary) for summary in summaries)
    assert summaries[0].lower < summaries[0].upper
    assert summaries[1].lower < summaries[1].upper
    # English scores are higher than Turkish ones, so posterior mean should reflect that.
    assert summaries[0].mean > summaries[1].mean
    assert aggregator.metric_name == "ddi"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_aggregate_language_scores_wrapper_matches_aggregator() -> None:
    records = _make_records()
    direct = aggregate_language_scores(
        records,
        draws=150,
        tune=150,
        chains=2,
        cores=1,
        random_seed=999,
    )
    manual_aggregator = HierarchicalAggregator(
        draws=150,
        tune=150,
        chains=2,
        cores=1,
        random_seed=999,
    )
    manual_aggregator.fit(records)
    manual = manual_aggregator.summaries()

    assert [s.language for s in direct] == [s.language for s in manual]
    for s_direct, s_manual in zip(direct, manual, strict=True):
        assert s_direct.language == s_manual.language
        assert pytest.approx(s_direct.mean, rel=0.15) == s_manual.mean
        assert pytest.approx(s_direct.lower, rel=0.20) == s_manual.lower
        assert pytest.approx(s_direct.upper, rel=0.20) == s_manual.upper
