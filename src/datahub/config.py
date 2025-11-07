"""Static configuration for dataset download and preprocessing paths."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TypedDict


class XLWSDConfig(TypedDict):
    file_id: str
    archive_name: str
    folder_name: str


class MCLWICConfig(TypedDict):
    base_url: str
    archives: Dict[str, str]
    folder_name: str


class XLWICConfig(TypedDict):
    dataset_id: str
    folder_name: str
    archive_url: str
    archive_name: str


# Default directories used by the Typer CLI; callers may override these.
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_PROCESSED_ROOT = Path("data/preprocess")

# ---------------------------------------------------------------------------
# Dataset-specific configuration payloads.

XL_WSD: XLWSDConfig = {
    "file_id": "19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0",
    "archive_name": "xl-wsd.zip",
    "folder_name": "xl-wsd",
}

MCL_WIC: MCLWICConfig = {
    "base_url": "https://raw.githubusercontent.com/SapienzaNLP/mcl-wic/master",
    "archives": {
        "all": "SemEval-2021_MCL-WiC_all-datasets.zip",
        "test-gold": "SemEval-2021_MCL-WiC_test-gold-data.zip",
        "trial": "SemEval-2021_MCL-WiC_trial.zip",
    },
    "folder_name": "mcl-wic",
}

XL_WIC: XLWICConfig = {
    "dataset_id": "pasinit/xlwic",
    "folder_name": "xlwic",
    "archive_url": "https://pilehvar.github.io/xlwic/data/xlwic_datasets.zip",
    "archive_name": "xlwic_datasets.zip",
}


__all__ = [
    "DEFAULT_RAW_ROOT",
    "DEFAULT_PROCESSED_ROOT",
    "XL_WSD",
    "MCL_WIC",
    "XL_WIC",
    "XLWSDConfig",
    "MCLWICConfig",
    "XLWICConfig",
]
