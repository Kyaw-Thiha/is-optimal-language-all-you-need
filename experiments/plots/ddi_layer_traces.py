"""Line plot helper for per-layer lemma scores."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd
import plotly.express as px

from src.models import ModelKey
from .save_config import PlotSaveDestinations

LemmaTrace = Mapping[int, float]
LanguageTrace = Mapping[str, LemmaTrace]
LanguageTraces = Mapping[str, LanguageTrace]

__all__ = ["LanguageTraces", "plot_language_traces"]

LEMMAS_PER_FIG = 8


def _chunked(items: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


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

    df = pd.DataFrame(rows).sort_values(["language", "lemma"])

    for language, lang_df in df.groupby("language"):
        lemma_ids = lang_df["lemma"].unique()
        total_chunks = (len(lemma_ids) + LEMMAS_PER_FIG - 1) // LEMMAS_PER_FIG

        for chunk_idx, subset in enumerate(_chunked(lemma_ids, LEMMAS_PER_FIG), start=1):
            chunk_df = lang_df[lang_df["lemma"].isin(subset)]
            title = (
                f"{model_name} – {language} lemmas {chunk_idx}/{total_chunks} "
                f"({', '.join(subset)})"
            )
            fig = px.line(
                chunk_df,
                x="layer",
                y="score",
                color="lemma",
                markers=True,
                title=title,
                labels={"layer": "Transformer layer", "score": "Probe accuracy"},
            )
            fig.update_yaxes(range=[0.0, 1.0])

            if save_to:
                chunk_dest = PlotSaveDestinations(
                    directory=save_to.directory / language,
                    slug=f"{save_to.slug}-{language}-part{chunk_idx}",
                    save_static=save_to.save_static,
                    save_html=save_to.save_html,
                )
                chunk_dest.ensure_dir()
                if chunk_dest.save_static:
                    fig.write_image(str(chunk_dest.png_path), engine="kaleido")
                if chunk_dest.save_html:
                    fig.write_html(
                        str(chunk_dest.html_path),
                        include_plotlyjs="cdn",
                        full_html=True,
                    )
            else:
                fig.show()
