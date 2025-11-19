from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Sequence, Tuple, TypeVar

SampleT = TypeVar("SampleT")
BucketKey = Tuple[str, str]


@dataclass(frozen=True)
class BucketPlan(Generic[SampleT]):
    """Lookup helpers describing how samples map to lemma buckets."""

    lookup: Dict[int, Tuple[BucketKey, int]]
    sizes: Dict[BucketKey, int]
    indices: Dict[BucketKey, List[int]]


def build_bucket_plan(
    samples: Sequence[SampleT],
    key_fn: Callable[[SampleT], BucketKey],
) -> BucketPlan[SampleT]:
    """Group samples by `key_fn` and build lookup structures for chunking."""
    buckets: Dict[BucketKey, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        buckets[key_fn(sample)].append(idx)

    lookup: Dict[int, Tuple[BucketKey, int]] = {}
    sizes: Dict[BucketKey, int] = {}
    for key, indices in buckets.items():
        sizes[key] = len(indices)
        for slot, sample_idx in enumerate(indices):
            lookup[sample_idx] = (key, slot)

    return BucketPlan(lookup=lookup, sizes=sizes, indices=dict(buckets))
