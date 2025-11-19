from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.datahub.loader import load_preprocessed
from src.models import load_model, ModelKey
from src.models.extraction import HiddenStateExtractor
from src.embeddings.token import Span
from src.pipelines import BucketKey, build_bucket_plan, run_chunked_forward
from src.probes.linear_logistic import LinearLogisticProbe, LinearLogisticProbeConfig
from src.metrics.ddi import compute_ddi, DDIConfig
from src.metrics.ddi_policy import FixedThresholdPolicy
from src.metrics.aggregation import LemmaMetricRecord, aggregate_language_scores

LemmaTraces = Dict[str, Dict[str, Dict[int, float]]]  # language -> lemma -> layer -> score
MAX_SENTENCES_PER_CHUNK = 50_000


def handle_ready_lemma(
    lemma_key: BucketKey,
    layer_features: List[torch.Tensor],
    labels_by_lemma: Dict[BucketKey, np.ndarray],
    lemma_traces: LemmaTraces,
    records: List[LemmaMetricRecord],
) -> None:
    """Probe a finished lemma bucket and record the DDI summary."""
    labels = labels_by_lemma.pop(lemma_key, None)
    if labels is None:
        return

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        print(f"[ddi] Skipping {lemma_key[0]}/{lemma_key[1]}: only one sense present even after combining splits.")
        return

    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
    except ValueError:
        print(f"[ddi] Skipping {lemma_key[0]}/{lemma_key[1]}: not enough samples for a test split.")
        return

    lemma_scores: Dict[int, float] = {}
    for layer_idx, features_tensor in enumerate(layer_features):
        features = features_tensor.numpy()
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        probe = LinearLogisticProbe(LinearLogisticProbeConfig(max_iter=1000))
        probe.fit(X_train, y_train)
        predicted = probe.predict(X_test)
        accuracy = float((predicted == y_test).mean())
        lemma_scores[layer_idx] = accuracy

    language, lemma = lemma_key
    lemma_traces[language][lemma] = lemma_scores

    ddi = compute_ddi(lemma_scores, config=DDIConfig(threshold_policy=FixedThresholdPolicy(0.7)))
    if ddi.layer is not None:
        records.append(
            LemmaMetricRecord(
                language=language,
                lemma=lemma,
                metric="ddi",
                value=float(ddi.layer),
            )
        )
    print(f"{language}/{lemma} → DDI layer={ddi.layer}")


def run_ddi_xlwsd(model_name: ModelKey, device: str = "cuda:0", batch_size: int = 256):
    print("[ddi] Starting DDI pipeline.")
    train_samples = list(load_preprocessed("xlwsd", split="train"))
    val_samples = list(load_preprocessed("xlwsd", split="validation"))
    samples = train_samples + val_samples
    print(f"[ddi] Loaded {len(samples)} samples (train + validation); starting forward pass with batch size {batch_size}.")
    model = load_model(model_name, device=device)
    extractor = HiddenStateExtractor(model, batch_size=batch_size, to_cpu=True)

    lemma_traces: LemmaTraces = defaultdict(lambda: defaultdict(dict))
    records: List[LemmaMetricRecord] = []
    languages = sorted({sample.language for sample in samples})

    for language in languages:
        lang_samples = [sample for sample in samples if sample.language == language]
        if not lang_samples:
            continue
        print(f"[ddi] Processing language '{language}' with {len(lang_samples)} samples.")

        # Group language-specific samples by (language, lemma) so each bucket can be probed independently.
        plan = build_bucket_plan(
            lang_samples,
            key_fn=lambda sample: (sample.language, sample.lemma),
        )
        if not plan.indices:
            continue

        labels_by_lemma: Dict[BucketKey, np.ndarray] = {}
        for key, indices in plan.indices.items():
            # Encode sense tags into integer labels per bucket.
            lemma_samples = [lang_samples[i] for i in indices]
            sense_tags = [cast(str, sample.sense_tag) for sample in lemma_samples]
            label_map = {tag: idx for idx, tag in enumerate(sorted(set(sense_tags)))}
            labels_by_lemma[key] = np.asarray([label_map[tag] for tag in sense_tags], dtype=int)

        # Collect raw texts and spans for the extractor; XL-WSD guarantees spans are present.
        texts = [sample.text_a for sample in lang_samples]
        spans: List[Span] = []
        for sample in lang_samples:
            span = sample.target_span
            if span is None:
                raise ValueError("XL-WSD spans should all be present.")
            spans.append(cast(Span, span))

        def on_ready(lemma_key: BucketKey, layer_tensors: List[torch.Tensor]) -> None:
            # Callback invoked by run_chunked_forward every time a lemma bucket is full.
            handle_ready_lemma(lemma_key, layer_tensors, labels_by_lemma, lemma_traces, records)

        run_chunked_forward(
            extractor,
            texts=texts,
            spans=spans,
            sample_lookup=plan.lookup,
            bucket_sizes=plan.sizes,
            chunk_size=MAX_SENTENCES_PER_CHUNK,
            on_ready=on_ready,
            desc=f"Forward pass ({language})",
        )

    print(f"[ddi] Finished probing; collected {len(records)} DDI records. Running aggregation ...")

    summaries = aggregate_language_scores(
        records,
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        random_seed=123,
    )
    print(f"[ddi] Aggregation complete ({len(summaries)} summaries).")

    return summaries, lemma_traces, records
