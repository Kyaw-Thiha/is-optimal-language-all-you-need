"""Model registry for loading benchmark backends.

This module centralizes the metadata needed to instantiate each model used in
the benchmarking suite.  Each entry describes the Hugging Face identifier,
architecture class, and preferred runtime options so that loader code can keep
architecture-specific logic declarative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ArchType = Literal["encoder_only", "decoder_only", "seq2seq"]
ModelKey = Literal["mbert", "xlmr", "mt5", "llama3", "labse", "minilm"]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a model that can be instantiated for inference."""

    name: str
    hf_id: str
    arch: ArchType
    dtype: Optional[str] = "bf16"
    use_bitsandbytes: bool = False
    tokenizer_kwargs: Optional[dict[str, object]] = None


REGISTRY: dict[ModelKey, ModelSpec] = {
    "mbert": ModelSpec(
        name="mbert",
        hf_id="bert-base-multilingual-cased",
        arch="encoder_only",
        dtype="fp16",
    ),
    "xlmr": ModelSpec(
        name="xlmr",
        hf_id="xlm-roberta-base",
        arch="encoder_only",
        dtype="fp16",
    ),
    "mt5": ModelSpec(
        name="mt5",
        hf_id="google/mt5-small",
        arch="seq2seq",
        dtype="fp16",
    ),
    "llama3": ModelSpec(
        name="llama3",
        hf_id="meta-llama/Meta-Llama-3-8B",
        arch="decoder_only",
        use_bitsandbytes=True,
        dtype="bf16",
    ),
    "labse": ModelSpec(
        name="labse",
        hf_id="sentence-transformers/LaBSE",
        arch="encoder_only",
        dtype="fp16",
    ),
    "minilm": ModelSpec(
        name="minilm",
        hf_id="sentence-transformers/all-MiniLM-L12-v2",
        arch="encoder_only",
        dtype="fp16",
    ),
}


def get_spec(key: ModelKey) -> ModelSpec:
    """Return the ModelSpec registered under ``key``."""
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model spec '{key}'. Available: {list(REGISTRY)}") from exc
