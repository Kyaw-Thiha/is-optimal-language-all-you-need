from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from .sense_sample import SenseSample
from .helpers import ensure_mapping, load_materialized, sample_to_record, safe_sequence, to_int


def preprocess_xlwsd(output_root: Path) -> None:
    """Materialize XL-WSD into the unified SenseSample layout."""
    raw = load_materialized("pasinit/xl-wsd")
    target_root = output_root / "xlwsd"
    target_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in raw.items():
        records: List[Dict[str, Any]] = []
        for idx, row in enumerate(split_dataset):
            data = ensure_mapping(row)
            sense_keys = safe_sequence(data.get("sense_keys"))
            sample = SenseSample(
                sample_id=f"xlwsd-{split_name}-{idx}",
                dataset_id="xlwsd",
                split=str(split_name),
                language=str(data.get("language")),
                text_a=str(data.get("context")),
                text_b=None,
                lemma=str(data.get("lemma")),
                target_span=(
                    to_int(data.get("target_start_char")),
                    to_int(data.get("target_end_char")),
                ),
                sense_tag=sense_keys[0] if sense_keys else None,
                same_sense=None,
            )
            records.append(sample_to_record(sample))
        Dataset.from_list(records).save_to_disk(target_root / str(split_name))


def preprocess_xlwic(output_root: Path) -> None:
    """Materialize XL-WiC into the unified SenseSample layout."""
    raw = load_materialized("pasinit/xl-wic")
    target_root = output_root / "xlwic"
    target_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in raw.items():
        records: List[Dict[str, Any]] = []
        for idx, row in enumerate(split_dataset):
            data = ensure_mapping(row)
            sample = SenseSample(
                sample_id=f"xlwic-{split_name}-{idx}",
                dataset_id="xlwic",
                split=str(split_name),
                language=str(data.get("language")),
                text_a=str(data.get("sentence1")),
                text_b=str(data.get("sentence2")),
                lemma=str(data.get("lemma")),
                target_span=None,
                sense_tag=None,
                same_sense=to_int(data.get("label")),
            )
            records.append(sample_to_record(sample))
        Dataset.from_list(records).save_to_disk(target_root / str(split_name))


def preprocess_mclwic(output_root: Path) -> None:
    """Materialize MCL-WiC into the unified SenseSample layout."""
    raw = load_materialized("mcl-wic/mcl_wic")
    target_root = output_root / "mclwic"
    target_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in raw.items():
        records: List[Dict[str, Any]] = []
        for idx, row in enumerate(split_dataset):
            data = ensure_mapping(row)
            sample = SenseSample(
                sample_id=f"mclwic-{split_name}-{idx}",
                dataset_id="mclwic",
                split=str(split_name),
                language=str(data.get("language")),
                text_a=str(data.get("sentence1")),
                text_b=str(data.get("sentence2")),
                lemma=str(data.get("lemma")),
                target_span=None,
                sense_tag=None,
                same_sense=to_int(data.get("label")),
            )
            records.append(sample_to_record(sample))
        Dataset.from_list(records).save_to_disk(target_root / str(split_name))


def preprocess_datasets(output_root: Path = Path("data/preprocess")) -> None:
    """Preprocess every dataset and save them under data/preprocess/."""
    output_root.mkdir(parents=True, exist_ok=True)
    preprocess_xlwsd(output_root)
    preprocess_xlwic(output_root)
    preprocess_mclwic(output_root)


if __name__ == "__main__":
    preprocess_datasets()
