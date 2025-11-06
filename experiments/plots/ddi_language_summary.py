"""Bar chart helper for language-level DDI summaries."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px

from src.metrics.aggregation.records import LanguageSummary
from src.models import ModelKey


def plot_language_ddi(summaries: Sequence[LanguageSummary], model_name: ModelKey) -> None:
    """Visualize aggregated DDI scores per language."""
    if not summaries:
        return

    df = pd.DataFrame(
        {
            "language": [summary.language for summary in summaries],
            "ddi_mean": [summary.mean for summary in summaries],
            "ddi_lower": [summary.lower for summary in summaries],
            "ddi_upper": [summary.upper for summary in summaries],
        }
    ).sort_values("ddi_mean")

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
