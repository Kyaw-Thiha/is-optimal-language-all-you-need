"""Base interfaces for model runners used in benchmarking.

The classes defined here normalize the interaction with different Hugging Face
models so the metric code can treat encoder-only, decoder-only, and seq2seq
architectures uniformly.  Subclasses are expected to provide the actual model
loading logic, while the base class handles tokenization, device placement, and
forward-pass bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeVar

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .helpers import DeviceLike, coerce_device, resolve_dtype
from .registry import ModelSpec


@dataclass
class ModelOutputs:
    """Standardized output payload returned by a model forward pass.

    Attributes
    ----------
    input_ids:
        Token IDs fed into the model (batch_size × seq_len).
    attention_mask:
        Mask used during the forward pass (1 for active tokens, 0 for padding).
    logits:
        Final-layer logits when the checkpoint exposes a language-model head;
        `None` for encoders without heads (e.g., LaBSE).
    encoder_hidden_states:
        Tuple of hidden-state tensors for encoder stacks; index 0 is the embedding
        layer. Present for encoder-only and seq2seq models.
    decoder_hidden_states:
        Tuple of hidden-state tensors for decoder stacks (embedding at index 0);
        populated for decoder-only and seq2seq models.
    extra:
        Optional dictionary carrying auxiliary outputs (e.g., attention weights,
        pooled outputs, cross-attention maps).
    """

    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    encoder_hidden_states: Optional[Sequence[torch.Tensor]]
    decoder_hidden_states: Optional[Sequence[torch.Tensor]]
    extra: Optional[dict[str, Any]] = None


T_BaseRunner = TypeVar("T_BaseRunner", bound="BaseModelRunner")


class BaseModelRunner:
    """Common runtime wrapper for Hugging Face models."""

    def __init__(
        self,
        spec: ModelSpec,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: DeviceLike,
        capture_attentions: bool = False,
    ) -> None:
        self.spec = spec
        device_obj = coerce_device(device)
        model_any: Any = model
        model_any.to(device=device_obj)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device_obj
        self.capture_attentions = capture_attentions
        self.model.eval()

    @classmethod
    def load_from_spec(
        cls: type[T_BaseRunner],
        spec: ModelSpec,
        device: DeviceLike,
        capture_attentions: bool = False,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
    ) -> T_BaseRunner:
        """Instantiate a runner from a ModelSpec."""

        resolved_tokenizer_kwargs = dict(spec.tokenizer_kwargs or {})
        if tokenizer_kwargs:
            resolved_tokenizer_kwargs.update(tokenizer_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, **resolved_tokenizer_kwargs)
        model_kwargs = cls._build_model_kwargs(spec)
        model = cls._load_model(spec, **model_kwargs)
        model.config.output_hidden_states = True

        return cls(
            spec=spec,
            model=model,
            tokenizer=tokenizer,
            device=device,
            capture_attentions=capture_attentions,
        )

    @staticmethod
    def _build_model_kwargs(spec: ModelSpec) -> dict[str, Any]:
        """Derive keyword arguments for the Hugging Face model loader."""

        kwargs: dict[str, Any] = {}
        dtype = resolve_dtype(spec.dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if spec.use_bitsandbytes:
            kwargs["load_in_8bit"] = True
        return kwargs

    @staticmethod
    def _load_model(spec: ModelSpec, **kwargs: Any) -> PreTrainedModel:
        """Create the underlying Hugging Face model (subclass responsibility)."""

        raise NotImplementedError("Subclasses must implement _load_model.")

    def tokenize(self, texts: Sequence[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize raw strings into padded tensors matching the model."""

        encoding = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            **kwargs,
        )
        return {k: v for k, v in encoding.items() if isinstance(v, torch.Tensor)}

    @torch.inference_mode()
    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutputs:
        """Run a forward pass and normalize the resulting tensors."""

        device_batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(
            **device_batch,
            output_hidden_states=True,
            output_attentions=self.capture_attentions,
            use_cache=False,
        )

        encoder_hidden_states: Optional[Sequence[torch.Tensor]]
        decoder_hidden_states: Optional[Sequence[torch.Tensor]]

        if self.spec.arch == "encoder_only":
            encoder_hidden_states = outputs.hidden_states or ()
            decoder_hidden_states = None
        elif self.spec.arch == "decoder_only":
            encoder_hidden_states = None
            decoder_hidden_states = outputs.hidden_states or ()
        else:  # seq2seq
            encoder_hidden_states = getattr(outputs, "encoder_hidden_states", None) or ()
            decoder_hidden_states = outputs.hidden_states or ()

        extra = {
            "attentions": getattr(outputs, "attentions", None),
            "cross_attentions": getattr(outputs, "cross_attentions", None),
            "encoder_attentions": getattr(outputs, "encoder_attentions", None),
            "pooler_output": getattr(outputs, "pooler_output", None),
        }

        # Remove empty extras to avoid clutter downstream.
        extra = {k: v for k, v in extra.items() if v is not None} or None

        if "input_ids" not in device_batch:
            raise ValueError("Tokenized batch must include 'input_ids'.")

        return ModelOutputs(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch.get("attention_mask"),
            logits=getattr(outputs, "logits", None),
            encoder_hidden_states=encoder_hidden_states,
            decoder_hidden_states=decoder_hidden_states,
            extra=extra,
        )
