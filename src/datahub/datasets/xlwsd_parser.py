"""
Parser utilities for XL-WSD archives extracted from the official release.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict


class XLWSDRawRow(TypedDict):
    language: str
    context: str
    lemma: str
    sense_keys: List[str]
    target_start_char: int
    target_end_char: int


_NO_SPACE_BEFORE = {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "'",
    '"',
    ")",
    "]",
    "}",
    "”",
    "’",
    "»",
    "''",
    "'s",
    "'re",
    "'ve",
    "'m",
    "'ll",
    "'d",
    "n't",
}
_NO_SPACE_AFTER = {"(", "[", "{", "``", "“", "«"}


def collect_xlwsd_rows(root: Path) -> Dict[str, List[XLWSDRawRow]]:
    """
    Traverse the XL-WSD archive and convert it into plain python rows grouped by split.
    """
    if not root.exists():
        raise FileNotFoundError(f"Missing XL-WSD directory at {root}")

    rows = {"train": [], "validation": [], "test": []}
    for split, language, data_path, gold_path in _iter_corpora(root):
        sense_map = _read_gold(gold_path)
        for context, spans, instances in _parse_sentence_instances(data_path):
            for inst in instances:
                span = spans.get(inst.identifier)
                if not span:
                    continue
                rows[split].append(
                    XLWSDRawRow(
                        language=language,
                        context=context,
                        lemma=inst.lemma or "",
                        sense_keys=sense_map.get(inst.identifier, []),
                        target_start_char=span[0],
                        target_end_char=span[1],
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# Internal helpers


def _iter_corpora(root: Path) -> Iterator[Tuple[str, str, Path, Path]]:
    training_root = root / "training_datasets"
    if training_root.exists():
        for corpus_dir in sorted(p for p in training_root.iterdir() if p.is_dir()):
            base = corpus_dir.name
            language = base.split("_")[-1].split("-")[-1].lower()
            data_path = corpus_dir / f"{base}.data.xml"
            gold_path = corpus_dir / f"{base}.gold.key.txt"
            if data_path.exists() and gold_path.exists():
                yield "train", language, data_path, gold_path

    eval_root = root / "evaluation_datasets"
    if not eval_root.exists():
        return

    for corpus_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        name = corpus_dir.name  # e.g., dev-ca or test-en
        if "-" not in name:
            continue
        prefix, lang = name.split("-", 1)
        language = lang.lower()
        split = "validation" if prefix == "dev" else "test" if prefix == "test" else None
        if not split:
            continue
        data_path = corpus_dir / f"{name}.data.xml"
        gold_path = corpus_dir / f"{name}.gold.key.txt"
        if data_path.exists() and gold_path.exists():
            yield split, language, data_path, gold_path


def _read_gold(path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.strip().split()
        if len(tokens) >= 2:
            mapping[tokens[0]] = tokens[1:]
    return mapping


@dataclass
class _Token:
    text: str
    identifier: Optional[str]
    lemma: Optional[str]
    is_instance: bool


def _parse_sentence_instances(path: Path) -> Iterator[Tuple[str, Dict[str, Tuple[int, int]], List[_Token]]]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    for sentence in root.iterfind(".//sentence"):
        tokens = _collect_tokens(sentence)
        if not tokens:
            continue
        context, spans = _reconstruct(tokens)
        instances = [tok for tok in tokens if tok.is_instance and tok.identifier]
        yield context, spans, instances


def _collect_tokens(sentence: ET.Element) -> List[_Token]:
    tokens: List[_Token] = []
    for child in sentence:
        if child.tag not in {"wf", "instance"}:
            continue
        text = (child.text or "").strip()
        if not text:
            continue
        tokens.append(
            _Token(
                text=text,
                identifier=child.get("id"),
                lemma=child.get("lemma"),
                is_instance=child.tag == "instance",
            )
        )
    return tokens


def _reconstruct(tokens: Sequence[_Token]) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    parts: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    offset = 0
    previous: Optional[str] = None

    for token in tokens:
        needs_space = bool(parts)
        if token.text in _NO_SPACE_BEFORE:
            needs_space = False
        if previous in _NO_SPACE_AFTER:
            needs_space = False
        if needs_space:
            parts.append(" ")
            offset += 1

        start = offset
        parts.append(token.text)
        offset += len(token.text)

        if token.identifier:
            spans[token.identifier] = (start, offset)
        previous = token.text

    return "".join(parts), spans


__all__ = ["XLWSDRawRow", "collect_xlwsd_rows"]
