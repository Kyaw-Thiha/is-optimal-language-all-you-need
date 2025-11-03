"""Unit tests for the model registry and runner stack."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.models import load_model
from src.models.base import BaseModelRunner
from src.models.decoder_only import DecoderModelRunner
from src.models.encoder_only import EncoderModelRunner
from src.models.helpers import DeviceLike, coerce_device, resolve_dtype
from src.models.registry import ModelSpec, get_spec
from src.models.seq2seq import Seq2SeqModelRunner


# ---------------------------------------------------------------------------
# Helper stubs


class DummyTokenizer:
    """Tokenizer stub that produces deterministic tensors."""

    def __call__(
        self,
        texts: Iterable[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        batch = len(list(texts))
        seq_len = 4
        input_ids = torch.arange(batch * seq_len, dtype=torch.long).reshape(batch, seq_len)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyModel:
    """Minimal stand-in for Hugging Face PreTrainedModel outputs."""

    def __init__(self, arch: str, *, return_logits: bool) -> None:
        self.arch = arch
        self.return_logits = return_logits
        self.config = SimpleNamespace(output_hidden_states=False)
        self.last_device: DeviceLike | None = None

    def to(self, *, device: torch.device) -> "DummyModel":
        self.last_device = device
        return self

    def eval(self) -> None:
        # Mirror PreTrainedModel API; nothing else required for tests.
        return None

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_hidden_states: bool,
        output_attentions: bool,
        use_cache: bool,
        **_: object,
    ) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        hidden_size = 8

        # Embedding layer + one transformer block for simplicity.
        hidden_states = tuple(
            torch.full((batch_size, seq_len, hidden_size), fill_value=float(idx))
            for idx in range(2)
        )
        logits = (
            torch.randn(batch_size, seq_len, 16)
            if self.return_logits
            else None
        )

        attentions = (
            (torch.zeros(batch_size, 1, seq_len, seq_len),)
            if output_attentions
            else None
        )

        if self.arch == "seq2seq":
            encoder_hidden_states = tuple(
                torch.full((batch_size, seq_len, hidden_size), fill_value=idx + 10.0)
                for idx in range(2)
            )
            cross_attentions = attentions
        else:
            encoder_hidden_states = None
            cross_attentions = None

        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
            encoder_attentions=None,
            pooler_output=torch.zeros(batch_size, hidden_size),
        )


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all tests use the dummy tokenizer to avoid network calls."""

    monkeypatch.setattr(
        "src.models.base.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )


def _patch_model_loader(monkeypatch: pytest.MonkeyPatch, arch: str, *, return_logits: bool) -> None:
    """Patch the appropriate AutoModel loader to hand back DummyModel."""

    dummy = DummyModel(arch, return_logits=return_logits)

    if arch == "encoder_only":
        monkeypatch.setattr(
            "src.models.encoder_only.AutoModelForMaskedLM.from_pretrained",
            lambda *args, **kwargs: dummy,
        )
        monkeypatch.setattr(
            "src.models.encoder_only.AutoModel.from_pretrained",
            lambda *args, **kwargs: dummy,
        )
    elif arch == "decoder_only":
        monkeypatch.setattr(
            "src.models.decoder_only.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: dummy,
        )
    elif arch == "seq2seq":
        monkeypatch.setattr(
            "src.models.seq2seq.AutoModelForSeq2SeqLM.from_pretrained",
            lambda *args, **kwargs: dummy,
        )
    else:
        raise ValueError(f"Unsupported arch for dummy model: {arch}")


# ---------------------------------------------------------------------------
# Helper tests


def test_resolve_dtype_roundtrips() -> None:
    assert resolve_dtype("fp16") is torch.float16
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype(None) is None


def test_resolve_dtype_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_dtype("float128")


def test_coerce_device_variants() -> None:
    assert coerce_device("cpu") == torch.device("cpu")
    assert coerce_device(torch.device("cpu")) == torch.device("cpu")
    assert coerce_device(0) == torch.device("cuda:0")


def test_get_spec_known_and_unknown() -> None:
    assert get_spec("mbert").arch == "encoder_only"
    with pytest.raises(ValueError):
        get_spec("does-not-exist")


# ---------------------------------------------------------------------------
# Runner behaviour


def test_encoder_runner_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        name="dummy-encoder",
        hf_id="dummy/encoder",
        arch="encoder_only",
        dtype="fp16",
        use_bitsandbytes=False,
    )
    _patch_model_loader(monkeypatch, "encoder_only", return_logits=True)

    runner = EncoderModelRunner.load_from_spec(spec, device="cpu")
    batch = runner.tokenize(["Hello world"])
    outputs = runner.forward(batch)

    assert outputs.encoder_hidden_states is not None
    assert len(outputs.encoder_hidden_states) == 2
    assert outputs.decoder_hidden_states is None
    assert outputs.logits is not None
    assert outputs.logits.shape[0:2] == outputs.input_ids.shape


def test_decoder_runner_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        name="dummy-decoder",
        hf_id="dummy/decoder",
        arch="decoder_only",
        dtype="bf16",
        use_bitsandbytes=False,
    )
    _patch_model_loader(monkeypatch, "decoder_only", return_logits=True)

    runner = DecoderModelRunner.load_from_spec(spec, device="cpu")
    batch = runner.tokenize(["Decoder run"])
    outputs = runner.forward(batch)

    assert outputs.encoder_hidden_states is None
    assert outputs.decoder_hidden_states is not None
    assert len(outputs.decoder_hidden_states) == 2
    assert outputs.logits is not None


def test_seq2seq_runner_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        name="dummy-seq2seq",
        hf_id="dummy/seq2seq",
        arch="seq2seq",
        dtype="fp16",
        use_bitsandbytes=False,
    )
    _patch_model_loader(monkeypatch, "seq2seq", return_logits=True)

    runner = Seq2SeqModelRunner.load_from_spec(spec, device="cpu", capture_attentions=True)
    batch = runner.tokenize(["Seq2Seq example"])
    outputs = runner.forward(batch)

    assert outputs.encoder_hidden_states is not None
    assert outputs.decoder_hidden_states is not None
    assert outputs.logits is not None
    assert "attentions" in (outputs.extra or {})


def test_runner_without_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        name="encoder-no-logits",
        hf_id="dummy/encoder-no-logits",
        arch="encoder_only",
        dtype=None,
        use_bitsandbytes=False,
    )
    _patch_model_loader(monkeypatch, "encoder_only", return_logits=False)

    runner = EncoderModelRunner.load_from_spec(spec, device="cpu")
    batch = runner.tokenize(["Embedding only"])
    outputs = runner.forward(batch)

    assert outputs.logits is None


def test_load_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        name="factory-decoder",
        hf_id="dummy/factory",
        arch="decoder_only",
        dtype="fp16",
        use_bitsandbytes=False,
    )
    registry_copy = {"factory-decoder": spec}

    monkeypatch.setattr("src.models.registry.REGISTRY", registry_copy, raising=False)
    monkeypatch.setattr("src.models.registry.get_spec", lambda key: registry_copy[key])
    _patch_model_loader(monkeypatch, "decoder_only", return_logits=True)

    runner = load_model("factory-decoder", device="cpu")

    assert isinstance(runner, BaseModelRunner)
    assert isinstance(runner, DecoderModelRunner)


def test_load_model_unknown_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        name="bad-arch",
        hf_id="dummy/bad",
        arch="encoder_only",
        dtype=None,
        use_bitsandbytes=False,
    )
    registry_copy = {"bad-arch": spec}

    monkeypatch.setattr("src.models.registry.REGISTRY", registry_copy, raising=False)
    monkeypatch.setattr("src.models.registry.get_spec", lambda key: registry_copy[key])
    monkeypatch.setattr("src.models.RUNNERS", {}, raising=False)

    with pytest.raises(ValueError):
        load_model("bad-arch")
