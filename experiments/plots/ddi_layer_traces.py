"""Line plot helper for per-layer lemma scores."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd
import plotly.express as px

from src.models import ModelKey
from .save_config import PlotSaveDestinations

LemmaTrace = Mapping[int, float]
LanguageTrace = Mapping[str, LemmaTrace]
LanguageTraces = Mapping[str, LanguageTrace]

__all__ = ["LanguageTraces", "plot_language_traces"]


def plot_language_traces(
    traces: LanguageTraces,
    model_name: ModelKey,
    save_to: Optional[PlotSaveDestinations] = None,
) -> None:
    """Plot per-layer probe scores for each lemma grouped by language."""
    if not traces:
        return

    rows: list[dict[str, object]] = []
    for language, lemmas in traces.items():
        for lemma, layer_scores in lemmas.items():
            for layer_idx, score in sorted(layer_scores.items()):
                rows.append(
                    {
                        "language": language,
                        "lemma": lemma,
                        "layer": layer_idx,
                        "score": score,
                    }
                )

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="layer",
        y="score",
        color="lemma",
        facet_row="language",
        markers=True,
        title=f"{model_name} – Probe accuracy per layer",
        labels={"layer": "Transformer layer", "score": "Probe accuracy"},
    )
    fig.update_yaxes(range=[0.0, 1.0])

    if save_to:
        save_to.ensure_dir()
        if save_to.save_static:
            fig.write_image(str(save_to.png_path), engine="kaleido")
        if save_to.save_html:
            fig.write_html(
                str(save_to.html_path),
                include_plotlyjs="cdn",
                full_html=True,
            )
    else:
        fig.show()
