"""Download helper for MCL-WiC bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from ..config import MCL_WIC
from ..io import (
    METADATA_SUFFIX,
    download_stream,
    needs_download,
    read_metadata,
    sha256sum,
    unzip,
    write_metadata,
)


def download_mclwic(raw_root: Path, splits: Iterable[str], force: bool = False) -> Path:
    """
    Download requested MCL-WiC bundles from the Sapienza repository.

    Parameters
    ----------
    raw_root:
        Directory used to store raw corpora (default: ``data/raw``).
    splits:
        Iterable of archive keys defined in ``config.MCL_WIC["archives"]``.
    force:
        If True, redownload archives even if they already exist locally.

    Returns
    -------
    Path
        Directory containing the extracted bundles (grouped by split key).
    """
    target_root = raw_root / MCL_WIC["folder_name"]
    target_root.mkdir(parents=True, exist_ok=True)

    archives: Dict[str, str] = MCL_WIC["archives"]  # type: ignore[assignment]
    for split in splits:
        if split not in archives:
            raise ValueError(f"Unknown MCL-WiC split '{split}'. Options: {tuple(archives)}")

        archive_name = archives[split]
        archive_path = target_root / archive_name
        meta_path = archive_path.with_suffix(METADATA_SUFFIX)
        meta = read_metadata(meta_path)
        expected_sha = meta.get("sha256")

        if force or needs_download(archive_path, expected_sha):
            url = f"{MCL_WIC['base_url']}/{archive_name}"
            print(f"Downloading MCL-WiC {split} from {url}")
            download_stream(url, archive_path)
            updated = dict(meta)
            updated["sha256"] = sha256sum(archive_path)
            write_metadata(meta_path, updated)
        else:
            print(f"MCL-WiC {split} archive present; skipping download.")

        split_dir = target_root / split
        print(f"Unpacking MCL-WiC {split} ...")
        unzip(archive_path, split_dir)

    return target_root


__all__ = ["download_mclwic"]
