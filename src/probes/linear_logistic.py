"""Scikit-learn backed logistic-regression probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import ArrayLike, BaseProbe, LabelLike, WeightsLike
from .helpers import ensure_1d_array, ensure_2d_array, ensure_sample_weights


@dataclass
class LinearLogisticProbeConfig:
    """Hyper-parameters forwarded to scikit-learn's LogisticRegression."""

    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    # Leave ``multi_class`` unset unless callers explicitly override it.
    multi_class: Optional[str] = None
    fit_intercept: bool = True
    class_weight: Optional[Union[str, Dict[int, float]]] = None
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    tol: float = 1e-4


class LinearLogisticProbe(BaseProbe):
    """Logistic regression probe."""

    def __init__(self, config: Optional[LinearLogisticProbeConfig] = None) -> None:
        self.config = config or LinearLogisticProbeConfig()
        self.model: Optional[LogisticRegression] = None

    def fit(
        self,
        features: ArrayLike,
        labels: LabelLike,
        sample_weights: WeightsLike = None,
    ) -> "LinearLogisticProbe":
        X = ensure_2d_array(features)
        y = ensure_1d_array(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature rows ({X.shape[0]}) and label count ({y.shape[0]}) must match")

        weights = ensure_sample_weights(sample_weights, X.shape[0])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kwargs = dict(
            penalty=self.config.penalty,
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            fit_intercept=self.config.fit_intercept,
            class_weight=self.config.class_weight,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            tol=self.config.tol,
        )
        if self.config.multi_class is not None:
            kwargs["multi_class"] = self.config.multi_class

        self.scaler = scaler
        self.model = LogisticRegression(**kwargs)
        self.model.fit(X_scaled, y, sample_weight=weights)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        X_scaled = self.scaler.transform(X)
        return model.predict(X_scaled)

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.stack([-decision, decision], axis=1)
        return _softmax(decision)

    def decision_function(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        if not hasattr(model, "decision_function"):
            raise AttributeError("Underlying model does not expose decision_function()")
        return model.decision_function(X)

    def _require_model(self) -> LogisticRegression:
        if self.model is None or not hasattr(self, "scaler"):
            raise RuntimeError("LinearLogisticProbe has not been fitted yet.")
        return self.model


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for fallback probability computation."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
