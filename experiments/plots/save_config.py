"""Shared configuration for storing Plotly figures on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlotSaveDestinations:
    """Resolved destinations for saving a single plot."""

    directory: Path
    slug: str
    save_static: bool
    save_html: bool

    def ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def png_path(self) -> Path:
        return self.directory / f"{self.slug}.png"

    @property
    def html_path(self) -> Path:
        return self.directory / f"{self.slug}.html"


@dataclass(frozen=True)
class PlotSaveConfig:
    """Factory for generating per-plot destinations under a structured folder."""

    base_dir: Path
    run_tag: str
    save_static: bool = True
    save_html: bool = True

    def for_plot(self, slug: str) -> PlotSaveDestinations:
        target = self.base_dir / self.run_tag
        return PlotSaveDestinations(
            directory=target,
            slug=slug,
            save_static=self.save_static,
            save_html=self.save_html,
        )


__all__ = ["PlotSaveConfig", "PlotSaveDestinations"]
