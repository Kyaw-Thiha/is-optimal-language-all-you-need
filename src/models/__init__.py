"""Factory helpers for instantiating model runners."""

from __future__ import annotations

from typing import Optional, Type

from .base import BaseModelRunner
from .decoder_only import DecoderModelRunner
from .encoder_only import EncoderModelRunner
from .helpers import DeviceLike
from .registry import ArchType, ModelKey, ModelSpec, REGISTRY, get_spec
from .seq2seq import Seq2SeqModelRunner

RunnerClass = Type[BaseModelRunner]

RUNNERS: dict[ArchType, RunnerClass] = {
    "encoder_only": EncoderModelRunner,
    "decoder_only": DecoderModelRunner,
    "seq2seq": Seq2SeqModelRunner,
}


def load_model(
    spec_key: ModelKey,
    device: DeviceLike = "cpu",
    capture_attentions: bool = False,
    tokenizer_kwargs: Optional[dict[str, object]] = None,
) -> BaseModelRunner:
    """Instantiate a model runner given a registry key."""
    spec: ModelSpec = get_spec(spec_key)

    try:
        runner_cls = RUNNERS[spec.arch]
    except KeyError as exc:
        raise ValueError(f"No runner registered for architecture '{spec.arch}'.") from exc

    return runner_cls.load_from_spec(
        spec=spec,
        device=device,
        capture_attentions=capture_attentions,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def list_available_models() -> tuple[str, ...]:
    """Return the registry keys for all configured models."""
    return tuple(sorted(REGISTRY))
