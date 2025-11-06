from collections import defaultdict
from typing import Dict, List, Tuple, cast

import numpy as np
from torch import Tensor

from src.datahub.loader import load_preprocessed
from src.models import load_model, ModelKey
from src.embeddings.token import Span, pool_token_embeddings
from src.probes.linear import LinearProbe, LinearProbeConfig
from src.metrics.ddi import compute_ddi, DDIConfig
from src.metrics.ddi_policy import FixedThresholdPolicy
from src.metrics.aggregation import LemmaMetricRecord, aggregate_language_scores

LemmaTraces = Dict[str, Dict[str, Dict[int, float]]]  # language -> lemma -> layer -> score


def run_ddi_xlwsd(model_name: ModelKey, device: str = "cuda:0"):
    samples = list(load_preprocessed("xlwsd", split="validation"))
    model = load_model(model_name, device=device)
    batch = model.tokenize([sample.text_a for sample in samples])
    outputs = model.forward(batch)

    layers: List[Tensor] = []
    if outputs.encoder_hidden_states:
        layers.extend(outputs.encoder_hidden_states)
    if outputs.decoder_hidden_states:
        layers.extend(outputs.decoder_hidden_states)

    # bucket samples by (language, lemma)
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        buckets[(sample.language, sample.lemma)].append(idx)

    lemma_traces: LemmaTraces = defaultdict(lambda: defaultdict(dict))
    records: list[LemmaMetricRecord] = []

    for (language, lemma), indices in buckets.items():
        # Sampling out the current lemma
        lemma_samples = [samples[i] for i in indices]

        # Building the labels
        sense_tags = [cast(str, sample.sense_tag) for sample in lemma_samples]
        label_map = {tag: idx for idx, tag in enumerate(sorted(set(sense_tags)))}
        labels = np.asarray([label_map[tag] for tag in sense_tags], dtype=int)

        # Building the spans
        target_spans = [sample.target_span for sample in lemma_samples]
        if any(span is None for span in target_spans):
            raise ValueError("XL-WSD spans should all be present.")
        target_spans = cast(List[Span], target_spans)

        lemma_scores: Dict[int, float] = {}

        for layer_idx, hidden_state in enumerate(layers):
            # Building the features
            layer_lemma = hidden_state[indices]
            features = pool_token_embeddings(layer_lemma, target_spans, "mean")

            # Carrying out the linear probing
            probe = LinearProbe(LinearProbeConfig(max_iter=200))
            probe.fit(features, labels)
            predicted = probe.predict(features)

            # Getting the score
            score = (predicted == labels).mean()
            lemma_scores[layer_idx] = score
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

    summaries = aggregate_language_scores(
        records,
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        random_seed=123,
    )

    return summaries, lemma_traces, records
