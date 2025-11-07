"""Download helper for XL-WiC via the official archive."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..config import XL_WIC
from ..io import (
    METADATA_SUFFIX,
    download_stream,
    needs_download,
    read_metadata,
    sha256sum,
    unzip,
    write_metadata,
)


def download_xlwic(raw_root: Path, configs: Iterable[str], force: bool = False) -> Path:
    """
    Download the XL-WiC archive once and extract it under ``data/raw/xlwic``.

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
        Directory containing the extracted XL-WiC payload.
    """
    cache_root = raw_root / XL_WIC["folder_name"]
    archive = raw_root / XL_WIC["archive_name"]
    cache_root.mkdir(parents=True, exist_ok=True)

    meta_path = archive.with_suffix(METADATA_SUFFIX)
    meta = read_metadata(meta_path)
    expected_sha = meta.get("sha256")

    if force or needs_download(archive, expected_sha):
        print("Downloading XL-WiC archive ...")
        try:
            download_stream(XL_WIC["archive_url"], archive)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download XL-WiC from the official mirror. "
                "Check your network connection or download the archive manually."
            ) from exc
        updated = dict(meta)
        updated["sha256"] = sha256sum(archive)
        write_metadata(meta_path, updated)
    else:
        print("XL-WiC archive present; skipping download.")

    marker = cache_root / "xlwic_datasets"
    if force or not marker.exists():
        print("Unpacking XL-WiC ...")
        unzip(archive, cache_root)
    else:
        print("XL-WiC already unpacked; skipping extraction.")

    return cache_root


__all__ = ["download_xlwic"]
