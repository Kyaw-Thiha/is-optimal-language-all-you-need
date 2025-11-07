from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Literal

from datasets import Dataset

from .config import DEFAULT_RAW_ROOT, XL_WIC
from .datasets.xlwic_utils import collect_xlwic_rows, normalize_xlwic_configs
from .sense_sample import SenseSample
from .helpers import ensure_mapping, load_materialized, sample_to_record, safe_sequence, to_int

DatasetId = Literal["xlwsd", "xlwic", "mclwic"]
ALL_DATASETS: Sequence[DatasetId] = ("xlwsd", "xlwic", "mclwic")


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


def preprocess_xlwic(
    output_root: Path,
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    configs: Sequence[str] = ("default",),
) -> None:
    """Materialize XL-WiC into the unified SenseSample layout."""
    dataset_root = raw_root / XL_WIC["folder_name"] / "xlwic_datasets"
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Missing XL-WiC archive under {dataset_root}. "
            "Run `python main.py datahub --xl-wic` to download it first."
        )

    languages = normalize_xlwic_configs(configs)
    raw_splits = collect_xlwic_rows(dataset_root, languages)
    if not any(raw_splits.values()):
        raise RuntimeError(
            "No XL-WiC samples found for the requested configurations. "
            f"Languages requested: {', '.join(languages)}."
        )

    target_root = output_root / "xlwic"
    target_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "validation", "test"):
        split_dataset = raw_splits.get(split_name)
        if not split_dataset:
            continue
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


def _normalize_selection(datasets: Optional[Sequence[DatasetId]]) -> Sequence[DatasetId]:
    """Return a deterministic, validated tuple of dataset ids."""
    if not datasets:
        return tuple(ALL_DATASETS)

    seen: Set[str] = set()
    normalized = []
    for dataset in datasets:
        if dataset not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset key '{dataset}'")
        if dataset in seen:
            continue
        normalized.append(dataset)
        seen.add(dataset)
    return tuple(normalized)


def preprocess_datasets(
    output_root: Path = Path("data/preprocess"),
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    datasets: Optional[Sequence[DatasetId]] = None,
    xlwic_configs: Sequence[str] = ("default",),
) -> None:
    """Preprocess selected datasets (default: all) and save them under data/preprocess/."""
    output_root.mkdir(parents=True, exist_ok=True)
    targets = _normalize_selection(datasets)

    for dataset in targets:
        if dataset == "xlwsd":
            preprocess_xlwsd(output_root)
        elif dataset == "xlwic":
            preprocess_xlwic(output_root, raw_root=raw_root, configs=xlwic_configs)
        elif dataset == "mclwic":
            preprocess_mclwic(output_root)
        else:
            raise ValueError(f"Unknown dataset key '{dataset}'")


if __name__ == "__main__":
    preprocess_datasets()
