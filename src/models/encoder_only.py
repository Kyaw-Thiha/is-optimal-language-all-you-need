"""Runner implementation for encoder-only transformer models."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel

from .base import BaseModelRunner
from .registry import ModelSpec


class EncoderModelRunner(BaseModelRunner):
    """Model runner specialization for encoder-only architectures."""

    @staticmethod
    def _load_model(spec: ModelSpec, **kwargs: Any) -> PreTrainedModel:
        """Instantiate the underlying encoder model."""

        if spec.arch != "encoder_only":
            raise ValueError(
                f"EncoderModelRunner expects an encoder-only ModelSpec, got '{spec.arch}'."
            )

        try:
            return AutoModelForMaskedLM.from_pretrained(spec.hf_id, **kwargs)
        except OSError:
            return AutoModel.from_pretrained(spec.hf_id, **kwargs)
