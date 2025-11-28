"""Optuna-backed hyperparameter tuning for the RandomForest probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

from .random_forest import RandomForestConfig, RandomForestProbe
from .tuning import ProbeTuner


@dataclass
class RandomForestTuningConfig:
    """Configuration controlling the global Optuna tuning pass."""

    trials: int = 25
    random_seed: int = 42
    validation_size: float = 0.2
    subsample: int = 2000  # cap tuning cost on huge buckets


class RandomForestOptunaTuner(ProbeTuner):
    """Run Optuna once and reuse the best config for every lemma."""

    def __init__(
        self,
        base_config: Optional[RandomForestConfig] = None,
        tuning_config: Optional[RandomForestTuningConfig] = None,
    ) -> None:
        self.base_config = base_config or RandomForestConfig()
        self.tuning_config = tuning_config or RandomForestTuningConfig()
        self._best_config: Optional[RandomForestConfig] = None
        self._study: Optional[optuna.Study] = None

    def tune(self, features: np.ndarray, labels: np.ndarray) -> None:
        if self._best_config is not None:
            return

        X, y = self._prepare_subset(features, labels)
        if np.unique(y).size < 2:
            self._best_config = self.base_config
            return

        sampler = optuna.samplers.TPESampler(seed=self.tuning_config.random_seed)
        self._study = optuna.create_study(direction="maximize", sampler=sampler, study_name="random_forest_probe")
        self._study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.tuning_config.trials)

        params = self._study.best_params
        self._best_config = RandomForestConfig(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            bootstrap=True,
            n_jobs=self.base_config.n_jobs,
            random_state=self.base_config.random_state,
        )

    def make_probe(self) -> RandomForestProbe:
        config = self._best_config or self.base_config
        return RandomForestProbe(config)

    # --- internals -----------------------------------------------------

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        config = RandomForestConfig(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_depth=trial.suggest_int("max_depth", 5, 32),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            bootstrap=True,
            n_jobs=self.base_config.n_jobs,
            random_state=self.base_config.random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.tuning_config.validation_size,
            random_state=self.tuning_config.random_seed,
            stratify=y,
        )
        probe = RandomForestProbe(config)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_val)
        return float((preds == y_val).mean())

    def _prepare_subset(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if features.shape[0] <= self.tuning_config.subsample:
            return features, labels
        rng = np.random.default_rng(self.tuning_config.random_seed)
        idx = rng.choice(features.shape[0], size=self.tuning_config.subsample, replace=False)
        return features[idx], labels[idx]
