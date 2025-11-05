"""LRU-style cache for frozen embedding batches."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Optional


class EmbeddingCache:
    """A minimal LRU cache keyed by hashable identifiers."""

    def __init__(self, max_items: int = 128) -> None:
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        self._store: OrderedDict[Hashable, Any] = OrderedDict()
        self._max_items = max_items

    def get(self, key: Hashable) -> Optional[Any]:
        """Return cached value for `key`, updating recency, or None if missing."""
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def set(self, key: Hashable, value: Any) -> None:
        """Insert value for `key`, evicting the oldest entry when capacity is exceeded."""
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self._max_items:
            self._store.popitem(last=False)
        self._store[key] = value

    def clear(self) -> None:
        """Remove every cached entry."""
        self._store.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


_GLOBAL_CACHE = EmbeddingCache()


def get_cached_embeddings(key: Hashable) -> Optional[Any]:
    """Module-level helper using a shared global cache instance."""
    return _GLOBAL_CACHE.get(key)


def set_cached_embeddings(key: Hashable, value: Any) -> None:
    """Store embedding tensors in the global cache."""
    _GLOBAL_CACHE.set(key, value)


def clear_cache() -> None:
    """Empty the global cache."""
    _GLOBAL_CACHE.clear()
