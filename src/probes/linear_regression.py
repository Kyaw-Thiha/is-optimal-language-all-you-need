"""Linear regression probe built on scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import ArrayLike, BaseProbe, LabelLike, WeightsLike
from .helpers import ensure_1d_array, ensure_2d_array


@dataclass
class LinearRegressionProbeConfig:
    fit_intercept: bool = True
    n_jobs: Optional[int] = None


class LinearRegressionProbe(BaseProbe):
    """Simple linear regressor for probing continuous targets."""

    def __init__(self, config: Optional[LinearRegressionProbeConfig] = None) -> None:
        self.config = config or LinearRegressionProbeConfig()
        self.model: Optional[LinearRegression] = None

    def fit(
        self,
        features: ArrayLike,
        labels: LabelLike,
        sample_weights: WeightsLike = None,
    ) -> "LinearRegressionProbe":
        X = ensure_2d_array(features)
        y = ensure_1d_array(labels)

        self.model = LinearRegression(
            fit_intercept=self.config.fit_intercept,
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        return model.predict(X)

    def _require_model(self) -> LinearRegression:
        if self.model is None:
            raise RuntimeError("LinearRegressionProbe has not been fitted yet.")
        return self.model


__all__ = ["LinearRegressionProbe", "LinearRegressionProbeConfig"]
