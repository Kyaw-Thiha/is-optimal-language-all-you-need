from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence, Tuple, cast

import datasets as hf_datasets

from .helpers import ensure_mapping
from .sense_sample import SenseSample

DatasetId = Literal["xlwsd", "xlwic", "mclwic"]
SplitName = Literal["train", "validation", "test"]


def load_preprocessed(
    dataset_id: DatasetId,
    split: SplitName,
    root: Path = Path("data/preprocess"),
) -> Iterator[SenseSample]:
    """Yield SenseSample records from the preprocessed cache."""
    dataset_path = root / dataset_id / split
    hf_split = hf_datasets.load_from_disk(str(dataset_path))

    for row in hf_split:
        data = ensure_mapping(row)
        raw_span = data.get("target_span")
        target_span: Optional[Tuple[int, int]] = None
        if raw_span is not None:
            span_seq = cast(Sequence[int], raw_span)
            target_span = (int(span_seq[0]), int(span_seq[1]))

        text_b_raw = data.get("text_b")
        sense_tag_raw = data.get("sense_tag")
        same_sense_raw = data.get("same_sense")

        yield SenseSample(
            sample_id=str(data.get("sample_id")),
            dataset_id=str(data.get("dataset_id")),
            split=str(data.get("split")),
            language=str(data.get("language")),
            text_a=str(data.get("text_a")),
            text_b=str(text_b_raw) if text_b_raw is not None else None,
            lemma=str(data.get("lemma")),
            target_span=target_span,
            sense_tag=str(sense_tag_raw) if sense_tag_raw is not None else None,
            same_sense=int(same_sense_raw) if same_sense_raw is not None else None,
        )
