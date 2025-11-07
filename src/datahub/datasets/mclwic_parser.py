"""
Parser utilities for the Sapienza MCL-WiC archives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, TypedDict


class MCLWiCRawRow(TypedDict):
    language: str
    sentence1: str
    sentence2: str
    lemma: str
    label: int


def collect_mclwic_rows(cache_root: Path, bundles: Sequence[str]) -> Dict[str, List[MCLWiCRawRow]]:
    """
    Aggregate rows from the requested MCL-WiC bundles.
    """
    splits: Dict[str, List[MCLWiCRawRow]] = {"train": [], "validation": [], "test": []}
    for key in bundles:
        base = cache_root / key / "MCL-WiC"
        if not base.exists():
            continue
        splits["train"].extend(_parse_partition(base / "training"))
        splits["validation"].extend(_parse_partition(base / "dev"))
        splits["test"].extend(_parse_partition(base / "test"))
    return splits


# ---------------------------------------------------------------------------
# Internal helpers


def _parse_partition(directory: Path) -> List[MCLWiCRawRow]:
    if not directory.exists():
        return []
    records: List[MCLWiCRawRow] = []
    for data_file in sorted(directory.rglob("*.data")):
        gold_file = data_file.with_suffix(".gold")
        if not gold_file.exists():
            continue
        gold_map = _load_gold(gold_file)
        data_entries = _load_json_array(data_file)
        language = _language_from_stem(data_file.stem)
        for entry in data_entries:
            sample_id = entry.get("id")
            if not sample_id:
                continue
            label = gold_map.get(sample_id)
            if label is None:
                continue
            records.append(
                MCLWiCRawRow(
                    language=language,
                    sentence1=str(entry.get("sentence1", "")),
                    sentence2=str(entry.get("sentence2", "")),
                    lemma=str(entry.get("lemma", "")),
                    label=label,
                )
            )
    return records


def _load_json_array(path: Path) -> List[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}, got {type(payload)}")
    return payload


def _load_gold(path: Path) -> Dict[str, int]:
    entries = _load_json_array(path)
    mapping: Dict[str, int] = {}
    for entry in entries:
        sample_id = entry.get("id")
        label = _tag_to_int(entry.get("tag"))
        if sample_id and label is not None:
            mapping[str(sample_id)] = label
    return mapping


def _tag_to_int(tag: Optional[str]) -> Optional[int]:
    if tag is None:
        return None
    token = str(tag).strip().upper()
    if token in {"T", "TRUE", "1"}:
        return 1
    if token in {"F", "FALSE", "0"}:
        return 0
    return None


def _language_from_stem(stem: str) -> str:
    parts = stem.split(".")
    if len(parts) >= 2:
        lang_pair = parts[1]
    else:
        lang_pair = parts[0]
    return lang_pair.split("-")[0].lower()


__all__ = ["MCLWiCRawRow", "collect_mclwic_rows"]
