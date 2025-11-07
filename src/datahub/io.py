"""Helpers for downloading archives and tracking cache metadata."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Optional

import requests

METADATA_SUFFIX = ".meta.json"


def sha256sum(path: Path) -> str:
    """Compute the SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: Path) -> Mapping[str, Any]:
    """Load metadata JSON attached to an archive, returning an empty mapping on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist metadata next to the archive to skip redundant downloads."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def needs_download(archive: Path, expected_sha: Optional[str]) -> bool:
    """Determine whether the archive must be re-downloaded."""
    if not archive.exists():
        return True
    if not expected_sha:
        return False
    return sha256sum(archive) != expected_sha


def download_stream(url: str, dest: Path) -> None:
    """Stream a remote file to disk atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
    os.replace(tmp.name, dest)


def run_gdown(file_id: str, dest: Path) -> None:
    """Invoke gdown via the current interpreter to fetch a Google Drive asset."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O",
            str(dest),
        ],
        check=True,
    )


def unzip(archive: Path, target_dir: Path) -> None:
    """Extract a zip archive to the provided directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target_dir)


__all__ = [
    "METADATA_SUFFIX",
    "download_stream",
    "needs_download",
    "read_metadata",
    "run_gdown",
    "sha256sum",
    "unzip",
    "write_metadata",
]
