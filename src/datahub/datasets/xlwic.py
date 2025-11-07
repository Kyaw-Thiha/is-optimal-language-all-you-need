"""Download helper for XL-WiC configs via the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from ..config import XL_WIC


def download_xlwic(raw_root: Path, configs: Iterable[str], force: bool = False) -> Path:
    """
    Fetch XL-WiC configurations and cache them locally with ``save_to_disk``.

    Parameters
    ----------
    raw_root:
        Directory used to store raw corpora (default: ``data/raw``).
    configs:
        Iterable of configuration names recognized by ``pasinit/xlwic``.
    force:
        If True, overwrite caches even when they already exist.

    Returns
    -------
    Path
        Directory containing cached dataset configs grouped by name.
    """
    cache_root = raw_root / XL_WIC["folder_name"]
    cache_root.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        target = cache_root / cfg
        if target.exists() and not force:
            print(f"XL-WiC {cfg} already cached; skipping.")
            continue

        print(f"Fetching XL-WiC config={cfg}")
        dataset_obj = load_dataset(XL_WIC["dataset_id"], cfg, streaming=False)
        if isinstance(dataset_obj, (IterableDataset, IterableDatasetDict)):
            raise TypeError("XL-WiC download requires eager (non-streaming) datasets")
        dataset_obj.save_to_disk(str(target))

    return cache_root


__all__ = ["download_xlwic"]
