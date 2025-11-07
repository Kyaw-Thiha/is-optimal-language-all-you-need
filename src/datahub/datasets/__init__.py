"""Dataset-specific download helpers used by the data hub pipeline."""

from __future__ import annotations

from .xlwsd import download_xlwsd
from .xlwic import download_xlwic
from .mclwic import download_mclwic

__all__ = [
    "download_xlwsd",
    "download_xlwic",
    "download_mclwic",
]
