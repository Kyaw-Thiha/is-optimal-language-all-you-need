"""Array conversion helpers shared across probe implementations."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from .base import ArrayLike, LabelLike, WeightsLike


def ensure_2d_array(features: ArrayLike, *, name: str = "features") -> np.ndarray:
    """Coerce features into a float64 numpy array of shape (n_samples, n_features)."""
    arr = _as_numpy(features, name=name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (batch × dim), got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def ensure_1d_array(values: LabelLike, *, name: str = "labels") -> np.ndarray:
    """Coerce labels into a 1-D numpy array."""
    arr = _as_numpy(values, name=name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (batch,), got shape {arr.shape}")
    return arr


def ensure_sample_weights(
    weights: WeightsLike,
    n_samples: int,
    name: str = "sample_weights",
) -> np.ndarray | None:
    """Validate optional sample weights."""
    if weights is None:
        return None
    arr = _as_numpy(weights, name=name)
    if arr.ndim != 1 or arr.shape[0] != n_samples:
        raise ValueError(f"{name} must be 1-D of length {n_samples}, got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _as_numpy(value: Union[np.ndarray, torch.Tensor, Sequence], *, name: str) -> np.ndarray:
    """Convert tensors/sequences to numpy arrays while keeping existing arrays intact."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr
