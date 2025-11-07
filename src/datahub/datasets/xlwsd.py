"""Download helper for XL-WSD."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..config import XL_WSD
from ..io import (
    METADATA_SUFFIX,
    needs_download,
    read_metadata,
    run_gdown,
    sha256sum,
    unzip,
    write_metadata,
)


def download_xlwsd(raw_root: Path, force: bool = False) -> Path:
    """
    Download the XL-WSD archive from Google Drive and extract it locally.

    Parameters
    ----------
    raw_root:
        Directory used to store raw corpora (default: ``data/raw``).
    force:
        If True, redownload the archive even when the checksum matches.

    Returns
    -------
    Path
        Directory where the archive was unpacked.
    """
    archive = raw_root / XL_WSD["archive_name"]
    target = raw_root / XL_WSD["folder_name"]
    target.parent.mkdir(parents=True, exist_ok=True)

    meta_path = archive.with_suffix(METADATA_SUFFIX)
    meta: Mapping[str, str] = read_metadata(meta_path)
    expected_sha = meta.get("sha256")

    if force or needs_download(archive, expected_sha):
        run_gdown(XL_WSD["file_id"], archive)
        updated = dict(meta)
        updated["sha256"] = sha256sum(archive)
        write_metadata(meta_path, updated)
    else:
        print("XL-WSD archive present; skipping download.")

    print("Unpacking XL-WSD ...")
    unzip(archive, target)
    return target


__all__ = ["download_xlwsd"]
