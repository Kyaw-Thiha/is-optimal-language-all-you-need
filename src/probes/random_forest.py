from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import ArrayLike, BaseProbe, LabelLike, WeightsLike
from .helpers import ensure_1d_array, ensure_2d_array, ensure_sample_weights


@dataclass
class RandomForestConfig:
    """Hyper-parameters forwarded to scikit-learn's RandomForestClassifier."""

    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, str]] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    class_weight: Optional[Union[str, Dict[Union[str, int], float]]] = None
    max_samples: Optional[Union[int, float]] = None
    ccp_alpha: float = 0.0

    def to_classifier_kwargs(self) -> Dict[str, Any]:
        """Return kwargs compatible with RandomForestClassifier."""
        return dict(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            max_samples=self.max_samples,
            ccp_alpha=self.ccp_alpha,
        )


class RandomForestProbe(BaseProbe):
    """Random Forest probe."""

    config: RandomForestConfig
    model: Optional[RandomForestClassifier]

    def __init__(self, config: Optional[RandomForestConfig] = None) -> None:
        self.config = config or RandomForestConfig()
        self.model = None

    def fit(
        self,
        features: ArrayLike,
        labels: LabelLike,
        sample_weights: WeightsLike = None,
    ) -> "RandomForestProbe":
        X = ensure_2d_array(features)
        y = ensure_1d_array(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature rows ({X.shape[0]}) and label count ({y.shape[0]}) must match")

        weights = ensure_sample_weights(sample_weights, X.shape[0])

        self.model = RandomForestClassifier(**self.config.to_classifier_kwargs())
        self.model.fit(X, y, sample_weight=weights)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        return np.asarray(model.predict(X))

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        if not hasattr(model, "predict_proba"):
            raise AttributeError("Underlying model does not expose predict_proba()")
        return np.asarray(model.predict_proba(X))

    def _require_model(self) -> RandomForestClassifier:
        if self.model is None:
            raise RuntimeError("RandomForestProbe has not been fitted yet.")
        return self.model
