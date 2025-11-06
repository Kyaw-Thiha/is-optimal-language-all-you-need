from collections import defaultdict
from typing import Dict, List, Tuple, cast

import numpy as np
from torch import Tensor

import pandas as pd
import plotly.express as px

from src.datahub.loader import load_preprocessed
from src.metrics.aggregation.records import LanguageSummary
from src.models import load_model, ModelKey
from src.embeddings.token import Span, pool_token_embeddings
from src.probes.linear import LinearProbe, LinearProbeConfig
from src.metrics.ddi import compute_ddi, DDIConfig
from src.metrics.ddi_policy import FixedThresholdPolicy
from src.metrics.aggregation import LemmaMetricRecord, aggregate_language_scores


def run_ddi_xlwsd(model_name: ModelKey, device: str = "cuda:0"):
    samples = list(load_preprocessed("xlwsd", split="validation"))
    model = load_model(model_name, device="cuda:0")
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

    records: list[LemmaMetricRecord] = []
    for (language, lemma), indices in buckets.items():
        # Building the labels
        sense_tags = [cast(str, sample.sense_tag) for sample in samples]
        label_map = {tag: idx for idx, tag in enumerate(sorted(set(sense_tags)))}
        labels = np.asarray([label_map[tag] for tag in sense_tags], dtype=int)

        # Building the spans
        target_spans = [sample.target_span for sample in samples]
        if any(span is None for span in target_spans):
            raise ValueError("XL-WSD spans should all be present.")
        target_spans = cast(List[Span], target_spans)

        lemma_scores: Dict[int, float] = {}

        for layer_idx, hidden_state in enumerate(layers):
            # Building the features
            layer_lemma = hidden_state[indices]
            features = pool_token_embeddings(hidden_state, target_spans, "mean")

            # Carrying out the linear probing
            probe = LinearProbe(LinearProbeConfig(max_iter=200))
            probe.fit(features, labels)
            predicted = probe.predict(features)

            # Getting the score
            score = (predicted == labels).mean()
            lemma_scores[layer_idx] = score

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
    plot_language_ddi(list(summaries), model_name)


def plot_language_ddi(summaries: list[LanguageSummary], model_name: ModelKey) -> None:
    df = pd.DataFrame(
        {
            "language": [s.language for s in summaries],
            "ddi_mean": [s.mean for s in summaries],
            "ddi_lower": [s.lower for s in summaries],
            "ddi_upper": [s.upper for s in summaries],
        }
    ).sort_values("ddi_mean")  # lower DDI → earlier disambiguation

    fig = px.bar(
        df,
        x="ddi_mean",
        y="language",
        orientation="h",
        error_x=df["ddi_upper"] - df["ddi_mean"],
        error_x_minus=df["ddi_mean"] - df["ddi_lower"],
        title=f"{model_name} – XL-WSD DDI per language",
        labels={"ddi_mean": "DDI (lower = earlier sense resolution)", "language": "Language"},
    )
    fig.update_layout(xaxis=dict(rangemode="tozero"))
    fig.show()
