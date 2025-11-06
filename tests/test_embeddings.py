"""Unit tests for pooling helpers in src.embeddings."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.token import gather_token_hidden_states, pool_token_embeddings, Span
from src.embeddings.sentence import pool_sentence_embeddings, pool_sentence_pair


# ---------------------------------------------------------------------------
# Token pooling


def test_gather_token_hidden_states_basic() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    spans: list[Span] = [(0, 2), (1, 3)]

    slices = gather_token_hidden_states(hidden, spans)

    assert len(slices) == 2
    assert slices[0].shape == (2, 4)
    assert torch.allclose(slices[0], hidden[0, 0:2])
    assert torch.allclose(slices[1], hidden[1, 1:3])


def test_gather_token_hidden_states_validates_shape() -> None:
    with pytest.raises(ValueError):
        gather_token_hidden_states(torch.randn(3, 4), [(0, 1)])  # missing hidden_size dimension


def test_gather_token_hidden_states_rejects_mismatch() -> None:
    hidden = torch.randn(2, 3, 4)
    with pytest.raises(ValueError):
        gather_token_hidden_states(hidden, [(0, 1)])


def test_gather_token_hidden_states_rejects_bad_span() -> None:
    hidden = torch.randn(1, 3, 4)
    with pytest.raises(ValueError):
        gather_token_hidden_states(hidden, [(2, 5)])


def test_pool_token_embeddings_mean_and_first() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    spans = [(0, 2), (1, 3)]

    mean_features = pool_token_embeddings(hidden, spans, strategy="mean")
    first_features = pool_token_embeddings(hidden, spans, strategy="first")

    assert mean_features.shape == (2, 2)
    assert torch.allclose(mean_features[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(mean_features[1], torch.tensor([10.0, 11.0]))

    assert torch.allclose(first_features[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(first_features[1], torch.tensor([9.0, 10.0]))


def test_pool_token_embeddings_invalid_strategy() -> None:
    hidden = torch.randn(1, 2, 3)
    with pytest.raises(ValueError):
        pool_token_embeddings(hidden, [(0, 1)], strategy="median")


# ---------------------------------------------------------------------------
# Sentence pooling


def test_pool_sentence_embeddings_cls_fallbacks_to_mean() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    pooled = pool_sentence_embeddings(hidden, attention_mask=mask, strategy="cls")

    expected = torch.tensor([[2.5, 2.5], [4.5, 4.5]])  # mean over valid tokens
    assert torch.allclose(pooled, expected)


def test_pool_sentence_embeddings_mean_with_mask() -> None:
    hidden = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    pooled = pool_sentence_embeddings(hidden, attention_mask=mask, strategy="mean")
    assert torch.allclose(pooled, torch.tensor([[3.0, 1.0]]))


def test_pool_sentence_embeddings_invalid_strategy() -> None:
    with pytest.raises(ValueError):
        pool_sentence_embeddings(torch.randn(1, 3, 4), strategy="max")  # type: ignore[arg-type]


def test_pool_sentence_pair_concatenates_outputs() -> None:
    hidden_a = torch.ones(2, 3, 2)
    hidden_b = 2 * torch.ones(2, 3, 2)

    pooled = pool_sentence_pair(hidden_a, hidden_b, strategy="mean")
    assert pooled.shape == (2, 4)
    assert torch.allclose(pooled[:, :2], torch.ones(2, 2))
    assert torch.allclose(pooled[:, 2:], 2 * torch.ones(2, 2))
