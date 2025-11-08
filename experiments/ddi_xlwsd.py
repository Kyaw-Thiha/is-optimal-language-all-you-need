from collections import defaultdict
from typing import Dict, List, Optional, Tuple, cast

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


def run_ddi_xlwsd(model_name: ModelKey, device: str = "cuda:0", batch_size: int = 256):
    print("[ddi] Starting DDI pipeline.")
    # Combine train + validation so each lemma has enough sense variety.
    train_samples = list(load_preprocessed("xlwsd", split="train"))
    val_samples = list(load_preprocessed("xlwsd", split="validation"))
    samples = train_samples + val_samples
    print(f"[ddi] Loaded {len(samples)} samples (train + validation); starting forward pass with batch size {batch_size}.")
    model = load_model(model_name, device=device)
    extractor = HiddenStateExtractor(model, batch_size=batch_size, to_cpu=True)

    lemma_traces: LemmaTraces = defaultdict(lambda: defaultdict(dict))
    records: list[LemmaMetricRecord] = []
    languages = sorted({sample.language for sample in samples})

    # Process one language at a time to keep the pooled tensors small.
    for language in languages:
        lang_samples = [sample for sample in samples if sample.language == language]
        if not lang_samples:
            continue
        print(f"[ddi] Processing language '{language}' with {len(lang_samples)} samples.")

        # ------------------------------------------------------------------
        # Bucket samples by (language, lemma) within this language.
        buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for local_idx, sample in enumerate(lang_samples):
            buckets[(sample.language, sample.lemma)].append(local_idx)
        if not buckets:
            continue

        # Prepare label storage and sample index lookup for this language.
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

        texts = [sample.text_a for sample in lang_samples]
        layer_count: int | None = None

        # ------------------------------------------------------------------
        # Stream batches for this language so only local tensors stay in memory.
        for chunk in extractor.iterate(texts, desc=f"Forward pass ({language})"):
            if layer_count is None:
                layer_count = len(chunk.hidden_states)
                print(f"[ddi] Forward pass complete for {language}; streaming features across {layer_count} layers.")
                for key, labels in labels_by_lemma.items():
                    length = len(labels)
                    feature_buffers[key] = [[None] * length for _ in range(layer_count)]

            chunk_spans: List[Span] = []
            for idx in chunk.indices:
                span = lang_samples[idx].target_span
                if span is None:
                    raise ValueError("XL-WSD spans should all be present.")
                chunk_spans.append(cast(Span, span))

            for layer_idx, layer_tensor in enumerate(chunk.hidden_states):
                pooled = pool_token_embeddings(layer_tensor, chunk_spans, "mean")
                for local_pos, local_sample_idx in enumerate(chunk.indices):
                    lemma_key, slot = sample_lookup[local_sample_idx]
                    feature_buffers[lemma_key][layer_idx][slot] = pooled[local_pos]

        if layer_count is None:
            print(f"[ddi] Skipping language '{language}': no layers extracted.")
            continue

        features_by_lemma: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        for key, layer_slots in feature_buffers.items():
            stacked_layers = []
            for layer_idx, slot_values in enumerate(layer_slots):
                if any(value is None for value in slot_values):
                    raise RuntimeError(f"Missing pooled features for {key} layer {layer_idx}.")
                stacked_layers.append(torch.stack(cast(List[torch.Tensor], slot_values), dim=0))
            features_by_lemma[key] = stacked_layers

        # ------------------------------------------------------------------
        # Probe each lemma in this language with a held-out split.
        for (language_key, lemma), labels in tqdm(labels_by_lemma.items(), desc=f"Probing lemmas ({language})"):
            layer_features = features_by_lemma[(language_key, lemma)]
            lemma_scores: Dict[int, float] = {}

            unique_labels = np.unique(labels)
            if unique_labels.size < 2:
                print(f"[ddi] Skipping {language_key}/{lemma}: only one sense present even after combining splits.")
                continue

            # Split lemma samples into train/test so probes evaluate on held-out contexts.
            try:
                train_idx, test_idx = train_test_split(
                    np.arange(len(labels)),
                    test_size=0.2,
                    random_state=42,
                    stratify=labels if unique_labels.size > 1 else None,
                )
            except ValueError:
                print(f"[ddi] Skipping {language_key}/{lemma}: not enough samples for a test split.")
                continue

            for layer_idx, features_tensor in enumerate(layer_features):
                # Pool features for this layer and train/test the probe.
                features = features_tensor.numpy()
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                probe = LinearLogisticProbe(LinearLogisticProbeConfig(max_iter=200))
                probe.fit(X_train, y_train)
                predicted = probe.predict(X_test)
                accuracy = float((predicted == y_test).mean())
                lemma_scores[layer_idx] = accuracy

            lemma_traces[language_key][lemma] = lemma_scores

            ddi = compute_ddi(lemma_scores, config=DDIConfig(threshold_policy=FixedThresholdPolicy(0.7)))
            if ddi.layer is not None:
                records.append(
                    LemmaMetricRecord(
                        language=language_key,
                        lemma=lemma,
                        metric="ddi",
                        value=float(ddi.layer),
                    )
                )
            print(f"{language_key}/{lemma} → DDI layer={ddi.layer}")

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
