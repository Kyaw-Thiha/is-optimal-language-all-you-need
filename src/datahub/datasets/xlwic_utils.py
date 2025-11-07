"""
Helpers for normalizing XL-WiC configs and parsing the official archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypedDict


class XLWiCRawRow(TypedDict):
    language: str
    sentence1: str
    sentence2: str
    lemma: str
    label: int


@dataclass(frozen=True)
class _XLWiCLanguageInfo:
    code: str
    folder: str
    extended_name: str


_XLWIC_LANGUAGE_TABLE: Mapping[str, _XLWiCLanguageInfo] = {
    "en": _XLWiCLanguageInfo("en", "wic_english", "english"),
    "bg": _XLWiCLanguageInfo("bg", "xlwic_wn", "bulgarian"),
    "zh": _XLWiCLanguageInfo("zh", "xlwic_wn", "chinese"),
    "hr": _XLWiCLanguageInfo("hr", "xlwic_wn", "croatian"),
    "da": _XLWiCLanguageInfo("da", "xlwic_wn", "danish"),
    "nl": _XLWiCLanguageInfo("nl", "xlwic_wn", "dutch"),
    "et": _XLWiCLanguageInfo("et", "xlwic_wn", "estonian"),
    "fa": _XLWiCLanguageInfo("fa", "xlwic_wn", "farsi"),
    "ja": _XLWiCLanguageInfo("ja", "xlwic_wn", "japanese"),
    "ko": _XLWiCLanguageInfo("ko", "xlwic_wn", "korean"),
    "it": _XLWiCLanguageInfo("it", "xlwic_wikt", "italian"),
    "fr": _XLWiCLanguageInfo("fr", "xlwic_wikt", "french"),
    "de": _XLWiCLanguageInfo("de", "xlwic_wikt", "german"),
}

_RAW_SPLIT_NAMES: Mapping[str, str] = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}

_DEFAULT_LANGUAGES: Tuple[str, ...] = tuple(sorted(_XLWIC_LANGUAGE_TABLE.keys()))


def available_languages() -> Tuple[str, ...]:
    """Return the supported XL-WiC language codes."""
    return _DEFAULT_LANGUAGES


def normalize_xlwic_configs(configs: Iterable[str]) -> Tuple[str, ...]:
    """
    Convert CLI config arguments into canonical language codes.

    Accepts lowercase language codes (e.g., ``fr``), Hugging Face-style configs
    such as ``xlwic_en_fr``, or the sentinel values ``default`` / ``all``.
    """
    requested: MutableMapping[str, None] = {}
    for cfg in configs:
        token = (cfg or "").strip().lower()
        if not token:
            continue
        if token in {"default", "all", "*"}:
            return _DEFAULT_LANGUAGES
        if token.startswith("xlwic_"):
            token = token.split("_")[-1]
        if token not in _XLWIC_LANGUAGE_TABLE:
            raise ValueError(f"Unknown XL-WiC config '{cfg}'. Supported languages: {', '.join(_DEFAULT_LANGUAGES)}.")
        requested[token] = None

    if not requested:
        return _DEFAULT_LANGUAGES
    return tuple(sorted(requested.keys()))


def collect_xlwic_rows(dataset_root: Path, languages: Sequence[str]) -> Dict[str, List[XLWiCRawRow]]:
    """
    Parse the extracted XL-WiC archive and return rows grouped by split.

    Parameters
    ----------
    dataset_root:
        Directory that contains the unzipped ``xlwic_datasets`` payload.
    languages:
        Normalized language codes (all lowercase) to include.
    """
    splits: Dict[str, List[XLWiCRawRow]] = {split: [] for split in _RAW_SPLIT_NAMES}
    for lang in languages:
        lang_rows = _load_language_rows(dataset_root, lang)
        for split_name, rows in lang_rows.items():
            splits[split_name].extend(rows)
    return splits


# ---------------------------------------------------------------------------
# Internal helpers


def _load_language_rows(dataset_root: Path, lang: str) -> Dict[str, List[XLWiCRawRow]]:
    info = _XLWIC_LANGUAGE_TABLE.get(lang)
    if not info:
        raise ValueError(f"Unsupported XL-WiC language '{lang}'")

    rows: Dict[str, List[XLWiCRawRow]] = {split: [] for split in _RAW_SPLIT_NAMES}
    for split_name, raw_split in _RAW_SPLIT_NAMES.items():
        data_path, gold_path = _resolve_split_paths(dataset_root, info, raw_split)
        if not data_path or not data_path.exists():
            continue
        data_lines = _read_data_lines(data_path)
        if gold_path:
            if not gold_path.exists():
                continue
            labels = _read_gold_labels(gold_path)
            if len(data_lines) != len(labels):
                raise ValueError(
                    f"Label count mismatch for {lang} {split_name}: "
                    f"{len(data_lines)} samples vs {len(labels)} labels."
                )
            for payload, label in zip(data_lines, labels):
                payload.append(label)
        rows[split_name].extend(_lines_to_rows(data_lines, lang))

    return {split: data for split, data in rows.items() if data}


def _resolve_split_paths(
    dataset_root: Path,
    info: _XLWiCLanguageInfo,
    raw_split: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    lang = info.code
    if info.folder == "wic_english":
        if raw_split == "test":
            return None, None
        base = dataset_root / info.folder
        return base / f"{raw_split}_{lang}.txt", None

    subdir = dataset_root / info.folder / f"{info.extended_name}_{lang}"
    if raw_split == "test":
        data = subdir / f"{lang}_{raw_split}_data.txt"
        gold = subdir / f"{lang}_{raw_split}_gold.txt"
        return data, gold
    return subdir / f"{lang}_{raw_split}.txt", None


def _read_data_lines(path: Path) -> List[List[str]]:
    lines: List[List[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            lines.append(raw_line.split("\t"))
    return lines


def _read_gold_labels(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if raw_line:
                labels.append(raw_line)
    return labels


def _lines_to_rows(lines: List[List[str]], lang: str) -> List[XLWiCRawRow]:
    rows: List[XLWiCRawRow] = []
    for payload in lines:
        if len(payload) < 9:
            continue
        try:
            label = int(payload[8])
        except ValueError:
            continue
        rows.append(
            XLWiCRawRow(
                language=lang,
                sentence1=payload[6],
                sentence2=payload[7],
                lemma=payload[0],
                label=label,
            )
        )
    return rows


__all__ = ["XLWiCRawRow", "available_languages", "collect_xlwic_rows", "normalize_xlwic_configs"]
