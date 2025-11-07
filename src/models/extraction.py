"""Utilities for batched hidden-state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from .base import BaseModelRunner


@dataclass
class ForwardBatch:
    """Container holding a single batched forward pass."""

    indices: List[int]
    hidden_states: Sequence[torch.Tensor]
    logits: torch.Tensor | None


class HiddenStateExtractor:
    """Run model forwards in batches and stage hidden states on CPU."""

    def __init__(self, runner: BaseModelRunner, batch_size: int = 256, to_cpu: bool = True) -> None:
        self.runner = runner
        self.batch_size = batch_size
        self.to_cpu = to_cpu

    def run(self, texts: Sequence[str]) -> List[torch.Tensor]:
        """Return concatenated hidden states for every layer."""

        buffers: List[List[torch.Tensor]] = []
        for chunk in self._iterate_batches(texts):
            if not buffers:
                buffers = [[] for _ in range(len(chunk.hidden_states))]
            for layer_idx, tensor in enumerate(chunk.hidden_states):
                buffers[layer_idx].append(tensor)
        return [torch.cat(parts, dim=0) for parts in buffers]

    def _iterate_batches(self, texts: Sequence[str]) -> Iterable[ForwardBatch]:
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            tokenized = self.runner.tokenize(batch_texts)
            outputs = self.runner.forward(tokenized)

            layers: List[torch.Tensor] = []
            if outputs.encoder_hidden_states:
                layers.extend(outputs.encoder_hidden_states)
            if outputs.decoder_hidden_states:
                layers.extend(outputs.decoder_hidden_states)

            if self.to_cpu:
                layers = [layer.to("cpu") for layer in layers]
                logits = outputs.logits.to("cpu") if outputs.logits is not None else None
            else:
                logits = outputs.logits

            yield ForwardBatch(
                indices=list(range(start, start + len(batch_texts))),
                hidden_states=layers,
                logits=logits,
            )

            del layers
            torch.cuda.empty_cache()


__all__ = ["HiddenStateExtractor", "ForwardBatch"]
