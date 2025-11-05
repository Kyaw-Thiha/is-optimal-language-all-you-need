# Embeddings Module

Utilities for turning `SenseSample` batches into fixed-size features ready for probes and metrics.

## Components

- `token.py`: Pools hidden states over token spans (e.g., the target word in XL-WSD). Provides:
  - `gather_token_hidden_states`: extract span slices from a `(batch, seq_len, hidden)` tensor.
  - `pool_token_embeddings`: collapse each span into a single vector via `mean` or `first`.
- `sentence.py`: Sentence-level pooling helpers for WiC-style data.
  - `pool_sentence_embeddings`: average tokens or take the first token (`CLS`) for single sentences.
 - `pool_sentence_pair`: pools both sentences independently and concatenates the results.
- `cache.py`: Lightweight LRU cache. Call `get_cached_embeddings` / `set_cached_embeddings` if you want to reuse frozen activations across metric runs.
- `__init__.py`: Re-exports the public helpers and exposes the `EmbeddingCache` class.

## Typical Usage

1. Load or preprocess samples via `src.datahub`.
2. Run your model (see `src.models` runners) and collect hidden states.
3. Choose the pooling helper:
   - Token span datasets (XL-WSD) → `pool_token_embeddings`.
   - Sentence-pair datasets (XL-WiC, MCL-WiC) → `pool_sentence_pair`.
4. Optionally memoize with the cache before training probes or computing metrics.

Each pooling function returns tensors shaped for the next stage (e.g., `batch × hidden_size` or `batch × 2·hidden_size`), so probe implementations can operate on consistent inputs.

## Code Samples

For token-based embeddings,
```python
from src.datahub.loader import load_preprocessed
from src.embeddings.token import pool_token_embeddings
from src.models import load_model

samples = list(load_preprocessed("xlwsd", split="validation"))
model = load_model("xlmr-base")

batch = model.tokenize(sample.text_a for sample in samples)
outputs = model.forward(batch, output_hidden_states=True)
hidden = outputs.hidden_states[8]  # layer of interest

token_spans = [
    model.map_char_span_to_tokens(sample.text_a, sample.target_span)
    for sample in samples
]
features = pool_token_embeddings(hidden, token_spans, strategy="mean")
```

For sentence-based embeddings,
```python
from src.embeddings.sentence import pool_sentence_pair

samples = list(load_preprocessed("xlwic", split="validation"))
model = load_model("xlmr-base")

batch = model.tokenize(
    [(sample.text_a, sample.text_b) for sample in samples],
    pair=True,
)
outputs = model.forward(batch, output_hidden_states=True)
hidden = outputs.hidden_states[6]

wic_features = pool_sentence_pair(
    hidden_states_a=hidden["sentence1"],
    hidden_states_b=hidden["sentence2"],
    attention_mask_a=batch.attention_mask_a,
    attention_mask_b=batch.attention_mask_b,
    strategy="cls",
)
```
