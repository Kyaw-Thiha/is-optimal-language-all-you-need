from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datahub.loader import load_preprocessed
from src.models import load_model, ModelKey
from src.models.extraction import HiddenStateExtractor
from src.embeddings.token import Span, pool_token_embeddings
from src.probes.linear_logistic import LinearLogisticProbe, LinearLogisticProbeConfig
from src.metrics.ddi import compute_ddi, DDIConfig
from src.metrics.ddi_policy import FixedThresholdPolicy
from src.metrics.aggregation import LemmaMetricRecord, aggregate_language_scores

LemmaTraces = Dict[str, Dict[str, Dict[int, float]]]  # language -> lemma -> layer -> score
MAX_SENTENCES_PER_CHUNK = 50_000


def process_chunk(
    extractor: HiddenStateExtractor,
    language: str,
    lang_samples: List,
    chunk_indices: Sequence[int],
    labels_by_lemma: Dict[Tuple[str, str], np.ndarray],
    feature_buffers: Dict[Tuple[str, str], List[List[Optional[torch.Tensor]]]],
    sample_lookup: Dict[int, Tuple[Tuple[str, str], int]],
    layer_count: Optional[int],
) -> Optional[int]:
    """Run a forward pass over a subset of sentences and fill pooling buffers."""
    chunk_texts = [lang_samples[i].text_a for i in chunk_indices]
    for batch in extractor.iterate(chunk_texts, desc=f"Forward pass ({language})"):
        if layer_count is None:
            layer_count = len(batch.hidden_states)
            for key, labels in labels_by_lemma.items():
                feature_buffers[key] = [[None] * len(labels) for _ in range(layer_count)]

        global_indices = [chunk_indices[i] for i in batch.indices]
        chunk_spans: List[Span] = []
        for idx in global_indices:
            span = lang_samples[idx].target_span
            if span is None:
                raise ValueError("XL-WSD spans should all be present.")
            chunk_spans.append(cast(Span, span))

        for layer_idx, layer_tensor in enumerate(batch.hidden_states):
            pooled = pool_token_embeddings(layer_tensor, chunk_spans, "mean")
            for local_pos, global_idx in enumerate(global_indices):
                lemma_key, slot = sample_lookup[global_idx]
                feature_buffers[lemma_key][layer_idx][slot] = pooled[local_pos]
    return layer_count


def finalize_ready_lemmas(
    ready_keys: Iterable[Tuple[str, str]],
    feature_buffers: Dict[Tuple[str, str], List[List[Optional[torch.Tensor]]]],
    labels_by_lemma: Dict[Tuple[str, str], np.ndarray],
    lemma_traces: LemmaTraces,
    records: List[LemmaMetricRecord],
) -> None:
    """Probe finished lemmas, record DDI, and free their buffers."""
    for lemma_key in ready_keys:
        layer_slots = feature_buffers.get(lemma_key)
        labels = labels_by_lemma.get(lemma_key)
        if layer_slots is None or labels is None:
            continue

        layer_features: List[torch.Tensor] = []
        for slot_values in layer_slots:
            if any(value is None for value in slot_values):
                break
            layer_features.append(torch.stack(cast(List[torch.Tensor], slot_values), dim=0))
        else:
            unique_labels = np.unique(labels)
            if unique_labels.size < 2:
                print(f"[ddi] Skipping {lemma_key[0]}/{lemma_key[1]}: only one sense present even after combining splits.")
            else:
                try:
                    train_idx, test_idx = train_test_split(
                        np.arange(len(labels)),
                        test_size=0.2,
                        random_state=42,
                        stratify=labels,
                    )
                except ValueError:
                    print(f"[ddi] Skipping {lemma_key[0]}/{lemma_key[1]}: not enough samples for a test split.")
                else:
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

        feature_buffers.pop(lemma_key, None)
        labels_by_lemma.pop(lemma_key, None)


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

        buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for local_idx, sample in enumerate(lang_samples):
            buckets[(sample.language, sample.lemma)].append(local_idx)
        if not buckets:
            continue

        labels_by_lemma: Dict[Tuple[str, str], np.ndarray] = {}
        feature_buffers: Dict[Tuple[str, str], List[List[Optional[torch.Tensor]]]] = {}
        sample_lookup: Dict[int, Tuple[Tuple[str, str], int]] = {}

        for key, indices in buckets.items():
            lemma_samples = [lang_samples[i] for i in indices]
            sense_tags = [cast(str, sample.sense_tag) for sample in lemma_samples]
            label_map = {tag: idx for idx, tag in enumerate(sorted(set(sense_tags)))}
            labels = np.asarray([label_map[tag] for tag in sense_tags], dtype=int)
            target_spans = [sample.target_span for sample in lemma_samples]

            if any(span is None for span in target_spans):
                raise ValueError("XL-WSD spans should all be present.")

            labels_by_lemma[key] = labels
            for pos, local_sample_idx in enumerate(indices):
                sample_lookup[local_sample_idx] = (key, pos)

        layer_count: Optional[int] = None

        if len(lang_samples) > MAX_SENTENCES_PER_CHUNK:
            all_idx = np.arange(len(lang_samples))
            for start in range(0, len(lang_samples), MAX_SENTENCES_PER_CHUNK):
                chunk_idx = list(all_idx[start : start + MAX_SENTENCES_PER_CHUNK])
                layer_count = process_chunk(
                    extractor,
                    language,
                    lang_samples,
                    chunk_idx,
                    labels_by_lemma,
                    feature_buffers,
                    sample_lookup,
                    layer_count,
                )
                ready = [
                    key
                    for key, layer_slots in feature_buffers.items()
                    if all(all(tensor is not None for tensor in slot) for slot in layer_slots)
                ]
                finalize_ready_lemmas(ready, feature_buffers, labels_by_lemma, lemma_traces, records)
            if layer_count is None:
                print(f"[ddi] Skipping language '{language}': no layers extracted.")
                continue

            remaining = list(feature_buffers.keys())
            finalize_ready_lemmas(remaining, feature_buffers, labels_by_lemma, lemma_traces, records)
            continue

        layer_count = process_chunk(
            extractor,
            language,
            lang_samples,
            list(range(len(lang_samples))),
            labels_by_lemma,
            feature_buffers,
            sample_lookup,
            layer_count,
        )
        if layer_count is None:
            print(f"[ddi] Skipping language '{language}': no layers extracted.")
            continue

        ready = list(feature_buffers.keys())
        finalize_ready_lemmas(ready, feature_buffers, labels_by_lemma, lemma_traces, records)

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
