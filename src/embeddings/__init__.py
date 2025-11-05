"""Utilities for generating pooled embeddings from transformer hidden states."""

from .cache import EmbeddingCache, clear_cache, get_cached_embeddings, set_cached_embeddings
from .sentence import pool_sentence_pair
from .token import pool_token_embeddings

__all__ = [
    "EmbeddingCache",
    "clear_cache",
    "get_cached_embeddings",
    "set_cached_embeddings",
    "pool_sentence_pair",
    "pool_token_embeddings",
]
