"""Sentence-level pooling helpers for WiC-style benchmarks."""

from __future__ import annotations

from typing import Literal, Optional

import torch

SentenceStrategy = Literal["cls", "mean"]


def pool_sentence_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    strategy: SentenceStrategy = "cls",
) -> torch.Tensor:
    """
    Collapse token-level hidden states into sentence embeddings.

    Parameters
    ----------
    hidden_states:
        Tensor shaped (batch, seq_len, hidden_size).
    attention_mask:
        Optional binary mask shaped (batch, seq_len); values of 1 indicate
        real tokens and 0 indicate padding.
    strategy:
        Pooling rule:
        - "cls": take the first token (index 0). If the attention mask blanks
                 that position, the function falls back to mean pooling.
        - "mean": masked mean over all non-padding tokens.

    Returns
    -------
    torch.Tensor
        Tensor shaped (batch, hidden_size) with pooled sentence embeddings.
    """
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape (batch, seq_len, hidden_size)")

    if strategy not in {"cls", "mean"}:
        raise ValueError(f"Unsupported sentence pooling strategy: {strategy}")

    if attention_mask is None:
        attention_mask = torch.ones(
            hidden_states.shape[:2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must match hidden_states batch and sequence dimensions")

    if strategy == "cls":
        cls_tokens = hidden_states[:, 0, :]
        cls_mask = attention_mask[:, 0]
        if torch.all(cls_mask > 0):
            return cls_tokens
        # Fall back to mean when CLS is masked out
        strategy = "mean"

    if strategy == "mean":
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        weighted = hidden_states * mask
        return weighted.sum(dim=1) / token_counts


def pool_sentence_pair(
    hidden_states_a: torch.Tensor,
    hidden_states_b: torch.Tensor,
    attention_mask_a: Optional[torch.Tensor] = None,
    attention_mask_b: Optional[torch.Tensor] = None,
    strategy: SentenceStrategy = "cls",
) -> torch.Tensor:
    """
    Pool a pair of sentences and concatenate their representations.

    Parameters
    ----------
    hidden_states_a, hidden_states_b:
        Tensor shaped (batch, seq_len, hidden_size) for each sentence.
    attention_mask_a, attention_mask_b:
        Optional masks for each sentence.
    strategy:
        Pooling rule propagated to both sentences.

    Returns
    -------
    torch.Tensor
        Tensor shaped (batch, hidden_size * 2) suitable for binary probes.
    """
    pooled_a = pool_sentence_embeddings(hidden_states_a, attention_mask_a, strategy=strategy)
    pooled_b = pool_sentence_embeddings(hidden_states_b, attention_mask_b, strategy=strategy)
    return torch.cat([pooled_a, pooled_b], dim=-1)
