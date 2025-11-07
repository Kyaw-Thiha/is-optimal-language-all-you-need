"""Token-level pooling utilities for word-sense probes."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

Span = Tuple[int, int]  # (start_idx, end_idx) in token space, end exclusive
_TRUNCATION_WARNING_EMITTED = False


def gather_token_hidden_states(
    hidden_states: torch.Tensor,
    spans: Sequence[Span],
) -> List[torch.Tensor]:
    """
    Extract hidden-state slices corresponding to per-sample token spans.

    Parameters
    ----------
    hidden_states:
        Tensor shaped (batch, seq_len, hidden_size) from a transformer layer.
    spans:
        One (start, end) pair per sample, with `end` exclusive.

    Returns
    -------
    List[torch.Tensor]
        For each sample, a tensor shaped (span_len, hidden_size).
    """
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape (batch, seq_len, hidden_size)")

    if len(spans) != hidden_states.size(0):
        raise ValueError("Number of spans must match the batch dimension")

    slices: List[torch.Tensor] = []
    seq_len = hidden_states.size(1)

    global _TRUNCATION_WARNING_EMITTED

    for span, sample_states in zip(spans, hidden_states):
        start, end = span
        if end > seq_len or start >= seq_len:
            if not _TRUNCATION_WARNING_EMITTED:
                print(
                    "[token] Warning: target span exceeds tokenized length; clamping to available tokens (logged once)."
                )
                _TRUNCATION_WARNING_EMITTED = True
            start = min(start, seq_len - 1)
            end = min(end, seq_len)
        if end <= start:
            end = min(start + 1, seq_len)
            start = max(end - 1, 0)
        slices.append(sample_states[start:end])

    return slices


def pool_token_embeddings(
    hidden_states: torch.Tensor,
    spans: Sequence[Span],
    strategy: str = "mean",
) -> torch.Tensor:
    """
    Pool span slices into fixed-size vectors.

    Parameters
    ----------
    hidden_states:
        Tensor shaped (batch, seq_len, hidden_size).
    spans:
        One span per sample (end exclusive).
    strategy:
        Pooling rule. Supported values:
        - "mean": average over the span tokens (default).
        - "first": take the first token vector in the span.

    Returns
    -------
    torch.Tensor
        Tensor shaped (batch, hidden_size) containing pooled span embeddings.
    """
    slices = gather_token_hidden_states(hidden_states, spans)
    pooled: List[torch.Tensor] = []

    for span_slice in slices:
        if strategy == "mean":
            pooled.append(span_slice.mean(dim=0))
        elif strategy == "first":
            pooled.append(span_slice[0])
        else:
            raise ValueError(f"Unsupported token pooling strategy: {strategy}")

    return torch.stack(pooled, dim=0)
