"""Utilities for batched hidden-state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Iterator, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

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

    def iterate(self, texts: Sequence[str], *, desc: str = "Forward pass") -> Iterator[ForwardBatch]:
        iterator = self._iterate_batches(texts)
        total = ceil(len(texts) / self.batch_size) if texts else 0
        for chunk in tqdm(iterator, total=total, desc=desc, leave=False):
            yield chunk

    def run(self, texts: Sequence[str]) -> List[torch.Tensor]:
        """Return concatenated hidden states for every layer."""

        buffers: List[List[torch.Tensor]] = []
        max_seq_len = 0

        for chunk in self.iterate(texts):
            if not buffers:
                buffers = [[] for _ in range(len(chunk.hidden_states))]
            if chunk.hidden_states:
                max_seq_len = max(max_seq_len, chunk.hidden_states[0].shape[1])
            for layer_idx, tensor in enumerate(chunk.hidden_states):
                buffers[layer_idx].append(tensor)

        if not buffers:
            return []

        padded_layers: List[torch.Tensor] = []
        for layer_tensors in buffers:
            padded = [
                tensor
                if tensor.shape[1] == max_seq_len
                else F.pad(tensor, (0, 0, 0, max_seq_len - tensor.shape[1]))
                for tensor in layer_tensors
            ]
            padded_layers.append(torch.cat(padded, dim=0))
        return padded_layers

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
