"""Lemma-level Disambiguation Depth Index utility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .ddi_policy import FixedThresholdPolicy, ThresholdPolicy


@dataclass(frozen=True)
class DDIResult:
    """Result of computing DDI for a single lemma."""

    layer: Optional[int]
    threshold: float
    scores: Dict[int, float]

    @property
    def found(self) -> bool:
        """True if at least one layer met the threshold."""
        return self.layer is not None


@dataclass
class DDIConfig:
    """Configuration for `compute_ddi`."""

    min_layer: int = 0
    max_layer: Optional[int] = None
    threshold_policy: ThresholdPolicy = field(default_factory=lambda: FixedThresholdPolicy(0.7))

    def validate(self) -> None:
        if self.max_layer is not None and self.max_layer < self.min_layer:
            raise ValueError("max_layer cannot be smaller than min_layer.")


def compute_ddi(layer_scores: Mapping[int, float], config: Optional[DDIConfig] = None) -> DDIResult:
    """Compute the Disambiguation Depth Index for a lemma.

    Args:
        layer_scores: Mapping from layer index to separability score S(L).
        config: Optional configuration overriding defaults.

    Returns:
        DDIResult describing the earliest layer that satisfies τ, or None if unmet.
    """

    if not layer_scores:
        raise ValueError("layer_scores must contain at least one entry.")

    cfg = config or DDIConfig()
    cfg.validate()

    filtered_scores = _filter_and_sort_scores(layer_scores, cfg.min_layer, cfg.max_layer)
    if not filtered_scores:
        raise ValueError("No scores remain after applying layer bounds.")

    tau = cfg.threshold_policy.derive(filtered_scores)
    ddi_layer = _find_first_layer(filtered_scores, tau)

    return DDIResult(layer=ddi_layer, threshold=tau, scores=dict(filtered_scores))


def _filter_and_sort_scores(
    layer_scores: Mapping[int, float], min_layer: int, max_layer: Optional[int]
) -> Dict[int, float]:
    sorted_items: Tuple[Tuple[int, float], ...] = tuple(sorted(layer_scores.items(), key=lambda item: item[0]))
    filtered: Dict[int, float] = {}

    for layer, score in sorted_items:
        if layer < min_layer:
            continue
        if max_layer is not None and layer > max_layer:
            continue
        if not np.isfinite(score):
            raise ValueError(f"Non-finite score encountered at layer {layer}.")
        filtered[layer] = float(score)
    return filtered


def _find_first_layer(scores: Mapping[int, float], threshold: float) -> Optional[int]:
    for layer, score in scores.items():
        if score >= threshold:
            return layer
    return None
