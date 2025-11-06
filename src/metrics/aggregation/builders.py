"""Dataset builders and PyMC model construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pymc as pm

from .records import LemmaMetricRecord


@dataclass(frozen=True)
class PriorConfig:
    """Prior hyper-parameters for the hierarchical model."""

    mu_mean: float = 0.0
    mu_sigma: float = 10.0
    language_scale: float = 1.0
    lemma_scale: float = 1.0
    obs_scale: float = 1.0

    def validate(self) -> None:
        if self.mu_sigma <= 0 or self.language_scale <= 0 or self.lemma_scale <= 0 or self.obs_scale <= 0:
            raise ValueError("Prior scales must be strictly positive.")


@dataclass(frozen=True)
class HierarchicalDataset:
    values: np.ndarray
    language_ids: np.ndarray
    lemma_ids: np.ndarray
    language_labels: Sequence[str]
    lemma_labels: Sequence[str]
    metric_name: str


def build_dataset(records: Iterable[LemmaMetricRecord]) -> HierarchicalDataset:
    """Convert lemma records into arrays ready for PyMC."""
    record_list = list(records)
    if not record_list:
        raise ValueError("No records supplied for aggregation.")

    metric_names = {record.metric for record in record_list}
    if len(metric_names) != 1:
        joined = ", ".join(sorted(metric_names))
        raise ValueError(f"Hierarchical pooling expects a single metric at a time, received: {joined}")
    metric_name = metric_names.pop()

    languages = sorted({record.language for record in record_list})
    lemmas = sorted({record.lemma for record in record_list})
    language_index = {language: idx for idx, language in enumerate(languages)}
    lemma_index = {lemma: idx for idx, lemma in enumerate(lemmas)}

    values = np.asarray([record.value for record in record_list], dtype=float)
    if values.ndim != 1:
        raise ValueError("Metric values must be a one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Metric values contain non-finite entries.")

    language_ids = np.asarray([language_index[record.language] for record in record_list], dtype=int)
    lemma_ids = np.asarray([lemma_index[record.lemma] for record in record_list], dtype=int)

    return HierarchicalDataset(
        values=values,
        language_ids=language_ids,
        lemma_ids=lemma_ids,
        language_labels=languages,
        lemma_labels=lemmas,
        metric_name=metric_name,
    )


def build_model(dataset: HierarchicalDataset, priors: PriorConfig) -> pm.Model:
    """Create the PyMC model for hierarchical lemma pooling."""
    priors.validate()
    coords = {"language": dataset.language_labels, "lemma": dataset.lemma_labels}
    with pm.Model(coords=coords) as model:
        mu_global = pm.Normal("mu_global", mu=priors.mu_mean, sigma=priors.mu_sigma)
        sigma_language = pm.HalfNormal("sigma_language", sigma=priors.language_scale)
        sigma_lemma = pm.HalfNormal("sigma_lemma", sigma=priors.lemma_scale)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=priors.obs_scale)

        mu_language = pm.Normal("mu_language", mu=mu_global, sigma=sigma_language, dims="language")
        beta_lemma = pm.Normal("beta_lemma", mu=0.0, sigma=sigma_lemma, dims="lemma")

        pm.Normal(
            "observations",
            mu=mu_language[dataset.language_ids] + beta_lemma[dataset.lemma_ids],
            sigma=sigma_obs,
            observed=dataset.values,
        )
    return model
