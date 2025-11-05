"""Common probe interfaces and shared typing aliases."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]]
LabelLike = Union[np.ndarray, torch.Tensor, Sequence[int], Sequence[str]]
WeightsLike = Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]]


class BaseProbe(Protocol):
    """Protocol describing the minimal surface area for probe implementations."""

    def fit(
        self,
        features: ArrayLike,
        labels: LabelLike,
        sample_weights: WeightsLike = None,
    ) -> "BaseProbe": ...

    def predict(self, features: ArrayLike) -> np.ndarray: ...

    def predict_proba(self, features: ArrayLike) -> np.ndarray: ...
