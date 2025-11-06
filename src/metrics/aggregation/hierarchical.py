"""Hierarchical pooling based on PyMC for language-level aggregation."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, cast

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr
from arviz import InferenceData

from .builders import PriorConfig, build_dataset, build_model
from .records import LanguageSummary, LemmaMetricRecord


class HierarchicalAggregator:
    """Fits a lemma-level hierarchical model and exposes language summaries."""

    def __init__(
        self,
        priors: Optional[PriorConfig] = None,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.9,
        chains: int = 4,
        cores: Optional[int] = None,
        credible_interval: float = 0.90,
    ) -> None:
        self.priors = priors or PriorConfig()
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.chains = chains
        self.cores = cores
        self.credible_interval = credible_interval
        self._idata: Optional[InferenceData] = None
        self._language_labels: Sequence[str] = ()
        self._metric_name: Optional[str] = None

    def fit(self, records: Iterable[LemmaMetricRecord]) -> None:
        """Fit the hierarchical model to the provided lemma records."""
        dataset = build_dataset(records)
        model = build_model(dataset, self.priors)
        with model:
            self._idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                target_accept=self.target_accept,
                chains=self.chains,
                cores=self.cores,
                return_inferencedata=True,
                progressbar=False,
            )
        self._language_labels = tuple(dataset.language_labels)
        self._metric_name = dataset.metric_name

    def summaries(self) -> Sequence[LanguageSummary]:
        """Return posterior summaries for each language."""
        if self._idata is None:
            raise RuntimeError("HierarchicalAggregator.fit() must be called before summaries().")
        if not 0 < self.credible_interval < 1:
            raise ValueError("credible_interval must fall within (0, 1).")

        posterior_group = getattr(self._idata, "posterior", None)
        if posterior_group is None or "mu_language" not in posterior_group:
            raise RuntimeError("Posterior does not contain the expected 'mu_language' variable.")

        posterior = cast(xr.Dataset, posterior_group)["mu_language"]
        mean_da = cast(xr.DataArray, posterior.mean(dim=("chain", "draw")))
        hdi_da = cast(xr.DataArray, az.hdi(posterior, hdi_prob=self.credible_interval))

        means = np.asarray(mean_da, dtype=float)
        interval = np.asarray(hdi_da, dtype=float)
        if means.ndim != 1 or interval.ndim != 2 or interval.shape[1] != 2:
            raise RuntimeError("Unexpected posterior shape when summarising language means.")
        if len(self._language_labels) != means.shape[0]:
            raise RuntimeError("Language label count does not match posterior mean shape.")

        results: list[LanguageSummary] = []
        for idx, language in enumerate(self._language_labels):
            results.append(
                LanguageSummary(
                    language=str(language),
                    mean=float(means[idx]),
                    lower=float(interval[idx, 0]),
                    upper=float(interval[idx, 1]),
                )
            )
        return results

    @property
    def metric_name(self) -> Optional[str]:
        """Metric identifier used during fitting, if retained."""
        return self._metric_name
