# Model Loading Stack

This package wraps Hugging Face checkpoints in thin, architecture-aware runners so the
benchmarking code can request logits, embeddings, and layer-wise hidden states without
scattering model-specific conditionals.

## Modules

- `registry.py`: houses the `ModelSpec` dataclass and a registry of proposal models
  (mBERT, XLM-R, mT5, Llama 3, LaBSE, MiniLM). Each spec records the Hugging Face
  identifier, architecture family, preferred dtype, and whether quantized loading is
  enabled.
- `helpers.py`: shared utilities such as `resolve_dtype` and `coerce_device` that
  translate user-friendly strings/ints into `torch.dtype` and `torch.device`.
- `base.py`: declares the `BaseModelRunner` interface and the `ModelOutputs` dataclass.
  Subclasses inherit tokenize/forward helpers and only override `_load_model`.
- `encoder_only.py`: implements `EncoderModelRunner`, attempting to load a masked-LM
  head when available and falling back to a plain encoder.
- `decoder_only.py`: implements `DecoderModelRunner` for causal language models,
  including optional bitsandbytes (4-/8-bit) loading.
- `seq2seq.py`: implements `Seq2SeqModelRunner` for encoder–decoder architectures like
  mT5.
- `__init__.py`: exposes the `load_model` factory and a `list_available_models` helper.

## Key Data Classes

### ModelSpec

```python
ModelSpec(
    name="llama3",
    hf_id="meta-llama/Meta-Llama-3-8B",
    arch="decoder_only",
    dtype="bf16",
    use_bitsandbytes=True,
    tokenizer_kwargs=None,
)
```

The spec drives loader choices: which runner subclass to use, preferred precision, and
whether to activate 8-bit quantized loading.

### ModelOutputs

```python
ModelOutputs(
    input_ids=tensor([[    1, 2354,  673,   318, 50256]]),
    attention_mask=tensor([[1, 1, 1, 1, 1]]),
    logits=tensor([[[ -9.3, ...,  4.1], ..., [ -7.8, ...,  3.6]]]),
    encoder_hidden_states=(
        tensor([[[ 0.00, ...,  0.02], ..., [ 0.01, ...,  0.00]]]),
        tensor([[[ 0.01, ...,  0.01], ..., [ 0.02, ..., -0.01]]]),
        ...
    ),
    decoder_hidden_states=None,
    extra={"attentions": tuple_of_layer_attn_maps},
)
```

For decoder-only models, the encoder stack is `None` and `decoder_hidden_states` holds
the layer outputs instead; seq2seq models populate both sequences and also include
cross-attention weights inside `extra`.

## Usage

```python
from src.models import load_model

runner = load_model("xlmr", device="cuda:0")
batch = runner.tokenize(["He deposited the check at the bank."])
outputs = runner.forward(batch)

hidden_states = outputs.encoder_hidden_states  # tuple with embeddings at index 0
logits = outputs.logits                        # optional for LaBSE/MiniLM
```

Call `list_available_models()` to see the registered keys. Each runner ensures
`model.config.output_hidden_states` is enabled so the metrics pipeline can perform
layer-wise probes without additional wiring.
