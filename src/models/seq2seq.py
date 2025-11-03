"""Runner implementation for encoder-decoder (seq2seq) transformer models."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForSeq2SeqLM, PreTrainedModel

from .base import BaseModelRunner
from .registry import ModelSpec


class Seq2SeqModelRunner(BaseModelRunner):
    """Model runner specialization for encoder-decoder architectures."""

    @staticmethod
    def _load_model(spec: ModelSpec, **kwargs: Any) -> PreTrainedModel:
        """Instantiate the underlying sequence-to-sequence model."""

        if spec.arch != "seq2seq":
            raise ValueError(
                f"Seq2SeqModelRunner expects a seq2seq ModelSpec, got '{spec.arch}'."
            )

        return AutoModelForSeq2SeqLM.from_pretrained(spec.hf_id, **kwargs)
