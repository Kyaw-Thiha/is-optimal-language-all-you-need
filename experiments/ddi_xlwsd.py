from collections import defaultdict
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from tqdm import tqdm

from src.datahub.loader import load_preprocessed
from src.models import load_model, ModelKey
from src.models.extraction import HiddenStateExtractor
from src.embeddings.token import Span, pool_token_embeddings
from src.probes.linear_logistic import LinearLogisticProbe, LinearLogisticProbeConfig
from src.probes.linear_regression import LinearRegressionProbe, LinearRegressionProbeConfig
from src.metrics.ddi import compute_ddi, DDIConfig
from src.metrics.ddi_policy import FixedThresholdPolicy
from src.metrics.aggregation import LemmaMetricRecord, aggregate_language_scores

LemmaTraces = Dict[str, Dict[str, Dict[int, float]]]  # language -> lemma -> layer -> score


def run_ddi_xlwsd(model_name: ModelKey, device: str = "cuda:0", batch_size: int = 256):
    print("[ddi] Starting DDI pipeline.")
    samples = list(load_preprocessed("xlwsd", split="validation"))
    print(f"[ddi] Loaded {len(samples)} samples; starting forward pass with batch size {batch_size}.")
    model = load_model(model_name, device=device)
    extractor = HiddenStateExtractor(model, batch_size=batch_size, to_cpu=True)

    # bucket samples by (language, lemma)
    print("[ddi] Starting lemma bucketing.")
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        buckets[(sample.language, sample.lemma)].append(idx)
    print(f"[ddi] Bucketing complete; {len(buckets)} language/lemma pairs to probe.")

    # Pre-compute labels/spans for each lemma and remember how to map sample idx -> lemma slot.
    labels_by_lemma: Dict[Tuple[str, str], np.ndarray] = {}
    feature_buffers: Dict[Tuple[str, str], List[List[Optional[torch.Tensor]]]] = {}
    sample_lookup: Dict[int, Tuple[Tuple[str, str], int]] = {}

    for key, indices in buckets.items():
        lemma_samples = [samples[i] for i in indices]
        sense_tags = [cast(str, sample.sense_tag) for sample in lemma_samples]
        label_map = {tag: idx for idx, tag in enumerate(sorted(set(sense_tags)))}
        labels = np.asarray([label_map[tag] for tag in sense_tags], dtype=int)
        target_spans = [sample.target_span for sample in lemma_samples]

        if any(span is None for span in target_spans):
            raise ValueError("XL-WSD spans should all be present.")

        labels_by_lemma[key] = labels
        for pos, sample_idx in enumerate(indices):
            sample_lookup[sample_idx] = (key, pos)

    texts = [sample.text_a for sample in samples]
    layer_count: int | None = None

    # Stream batches through the model so only one batch touches GPU memory at a time.
    for chunk in extractor.iterate(texts):
        if layer_count is None:
            layer_count = len(chunk.hidden_states)
            print(f"[ddi] Forward pass complete; streaming features across {layer_count} layers.")
            for key, labels in labels_by_lemma.items():
                length = len(labels)
                feature_buffers[key] = [[None] * length for _ in range(layer_count)]

        # Gather the target spans for this chunk (they stay on CPU).
        chunk_spans: List[Span] = []
        for idx in chunk.indices:
            span = samples[idx].target_span
            if span is None:
                raise ValueError("XL-WSD spans should all be present.")
            chunk_spans.append(cast(Span, span))

        # Pool span embeddings layer by layer and stash them in the lemma buffers.
        for layer_idx, layer_tensor in enumerate(chunk.hidden_states):
            pooled = pool_token_embeddings(layer_tensor, chunk_spans, "mean")
            for local_pos, global_idx in enumerate(chunk.indices):
                lemma_key, slot = sample_lookup[global_idx]
                feature_buffers[lemma_key][layer_idx][slot] = pooled[local_pos]

    if layer_count is None:
        raise RuntimeError("No layers extracted from the model; aborting DDI run.")

    # Stack per-layer features for each lemma once everything has streamed through.
    features_by_lemma: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    for key, layer_slots in feature_buffers.items():
        stacked_layers = []
        for layer_idx, slot_values in enumerate(layer_slots):
            if any(value is None for value in slot_values):
                raise RuntimeError(f"Missing pooled features for {key} layer {layer_idx}.")
            stacked_layers.append(torch.stack(cast(List[torch.Tensor], slot_values), dim=0))
        features_by_lemma[key] = stacked_layers

    lemma_traces: LemmaTraces = defaultdict(lambda: defaultdict(dict))
    records: list[LemmaMetricRecord] = []

    print("[ddi] Starting lemma probing loop.")
    for (language, lemma), labels in tqdm(labels_by_lemma.items(), desc="Probing lemmas"):
        layer_features = features_by_lemma[(language, lemma)]
        lemma_scores: Dict[int, float] = {}

        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            print(f"[ddi] Skipping {language}/{lemma}: only one sense present in validation split.")
            continue

        for layer_idx, features_tensor in enumerate(layer_features):
            features = features_tensor.numpy()
            if unique_labels.size == 2:
                probe = LinearLogisticProbe(LinearLogisticProbeConfig(max_iter=200))
                probe.fit(features, labels)
                predicted = probe.predict(features)
                error = 1.0 - float((predicted == labels).mean())
            else:
                probe = LinearRegressionProbe(LinearRegressionProbeConfig())
                probe.fit(features, labels)
                predicted = np.rint(probe.predict(features)).astype(int)
                predicted = np.clip(predicted, unique_labels.min(), unique_labels.max())
                error = 1.0 - float((predicted == labels).mean())
            lemma_scores[layer_idx] = error

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
