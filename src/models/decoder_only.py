"""Runner implementation for decoder-only language models."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, PreTrainedModel

from .base import BaseModelRunner
from .registry import ModelSpec


class DecoderModelRunner(BaseModelRunner):
    """Model runner specialization for decoder-only architectures."""

    @staticmethod
    def _load_model(spec: ModelSpec, **kwargs: Any) -> PreTrainedModel:
        """Instantiate the underlying causal language model."""

        if spec.arch != "decoder_only":
            raise ValueError(
                f"DecoderModelRunner expects a decoder-only ModelSpec, got '{spec.arch}'."
            )

        load_kwargs = dict(kwargs)

        if spec.use_bitsandbytes:
            load_kwargs.setdefault("device_map", "auto")
            # BitsAndBytes loaders require skipping safetensors check in some cases.
            load_kwargs.setdefault("trust_remote_code", False)

        return AutoModelForCausalLM.from_pretrained(spec.hf_id, **load_kwargs)
