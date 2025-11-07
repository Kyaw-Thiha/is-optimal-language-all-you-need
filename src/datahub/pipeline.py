"""High-level orchestration for downloading and preprocessing datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Tuple, cast

from .datasets import download_mclwic, download_xlwsd, download_xlwic
from .preprocess import preprocess_datasets

DatasetId = Literal["xlwsd", "xlwic", "mclwic"]
ALL_DATASETS: Tuple[DatasetId, ...] = ("xlwsd", "xlwic", "mclwic")


@dataclass(frozen=True)
class DataRequest:
    """Describe which datasets to materialize and how to parameterize them."""

    datasets: Tuple[DatasetId, ...]
    xlwic_configs: Tuple[str, ...] = ("default",)
    mclwic_splits: Tuple[str, ...] = ("all",)

    @classmethod
    def from_flags(
        cls,
        all: bool,
        xl_wsd: bool,
        xl_wic: bool,
        mcl_wic: bool,
        xlwic_config: Sequence[str],
        mclwic_splits: Sequence[str],
    ) -> "DataRequest":
        """Translate CLI flags into a normalized request."""
        if all:
            selected = list(ALL_DATASETS)
        else:
            selected = [dataset for dataset, flag in zip(ALL_DATASETS, (xl_wsd, xl_wic, mcl_wic)) if flag]

        if not selected:
            raise ValueError("Select at least one dataset via --all or dataset flags.")

        configs = tuple(xlwic_config or ("default",))
        splits = tuple(mclwic_splits or ("all",))
        dataset_tuple = cast(Tuple[DatasetId, ...], tuple(selected))
        return cls(dataset_tuple, configs, splits)


def prepare_datasets(
    request: DataRequest,
    raw_root: Path,
    processed_root: Path,
    force: bool = False,
) -> None:
    """
    Run the full data pipeline: download corpora, then convert into SenseSample caches.
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    for dataset in request.datasets:
        if dataset == "xlwsd":
            download_xlwsd(raw_root, force=force)
        elif dataset == "xlwic":
            download_xlwic(raw_root, request.xlwic_configs, force=force)
        elif dataset == "mclwic":
            download_mclwic(raw_root, request.mclwic_splits, force=force)
        else:
            raise ValueError(f"Unknown dataset key '{dataset}'")

    preprocess_datasets(
        output_root=processed_root,
        raw_root=raw_root,
        datasets=request.datasets,
        xlwic_configs=request.xlwic_configs,
        mclwic_splits=request.mclwic_splits,
    )


__all__ = ["ALL_DATASETS", "DataRequest", "prepare_datasets"]
