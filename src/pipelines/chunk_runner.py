from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

import torch

from src.embeddings.token import Span, pool_token_embeddings
from src.models.extraction import HiddenStateExtractor

from .bucketing import BucketKey


def run_chunked_forward(
    extractor: HiddenStateExtractor,
    texts: Sequence[str],
    spans: Sequence[Span],
    sample_lookup: Dict[int, Tuple[BucketKey, int]],
    bucket_sizes: Dict[BucketKey, int],
    chunk_size: int,
    on_ready: Callable[[BucketKey, List[torch.Tensor]], None],
    desc: str = "Forward pass",
) -> None:
    """Stream batches through the model and call `on_ready` whenever a bucket fills."""
    if len(texts) != len(spans):
        raise ValueError("Texts and spans collections must be aligned.")

    layer_buffers: Dict[BucketKey, List[List[Optional[torch.Tensor]]]] = {}
    layer_count: Optional[int] = None

    for start in range(0, len(texts), chunk_size):
        chunk_indices = list(range(start, min(start + chunk_size, len(texts))))
        chunk_texts = [texts[i] for i in chunk_indices]

        for batch in extractor.iterate(chunk_texts, desc=desc):
            if layer_count is None:
                # Initialize per-layer buffers lazily once the first batch clarifies how many hidden-state tensors each forward pass produces.
                layer_count = len(batch.hidden_states)
                for key, size in bucket_sizes.items():
                    layer_buffers[key] = [[None] * size for _ in range(layer_count)]

            # Translate batch-local positions back to original sample indices and spans.
            global_indices = [chunk_indices[i] for i in batch.indices]
            chunk_spans = [spans[i] for i in global_indices]
            if any(span is None for span in chunk_spans):
                raise ValueError("All samples must provide a target span.")

            for layer_idx, layer_tensor in enumerate(batch.hidden_states):
                pooled = pool_token_embeddings(layer_tensor, chunk_spans, "mean")
                for local_pos, sample_idx in enumerate(global_indices):
                    bucket_key, slot = sample_lookup[sample_idx]
                    layer_buffers[bucket_key][layer_idx][slot] = pooled[local_pos]

            ready_keys: List[BucketKey] = []
            for key, slots in list(layer_buffers.items()):
                # Buckets are considered "ready" only when every slot across every layer contains a pooled embedding.
                bucket_complete = True
                for slot in slots:
                    if any(vec is None for vec in slot):
                        bucket_complete = False
                        break
                if bucket_complete:
                    ready_keys.append(key)

            for key in ready_keys:
                layer_slots = layer_buffers.pop(key)
                stacked_layers: List[torch.Tensor] = []
                for slot in layer_slots:
                    # Safe cast: readiness check above guarantees that no slot entry is None.
                    tensors = [cast(torch.Tensor, tensor) for tensor in slot]
                    stacked_layers.append(torch.stack(tensors, dim=0))

                # Notify the caller that all samples for this lemma are available.
                on_ready(key, stacked_layers)
