from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, cast


def sample_to_record(sample: Any) -> Dict[str, Any]:
    """Convert a dataclass-like record to a plain dict with list span."""
    from dataclasses import asdict

    record = cast(Dict[str, Any], asdict(sample))
    target_span = record.get("target_span")
    if target_span is not None:
        record["target_span"] = list(cast(Sequence[int], target_span))
    return record


def ensure_mapping(row: Any) -> Mapping[str, Any]:
    """Guarantee dataset rows behave like mappings."""
    if isinstance(row, Mapping):
        return row
    if isinstance(row, dict):
        return row
    raise TypeError(f"Unexpected row type: {type(row)}")


def safe_sequence(value: Any) -> Sequence[str]:
    """Return a sequence of strings even when the source is None or scalar."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def to_int(value: Any) -> int:
    """Robustly convert Hugging Face fields to ints."""
    if value is None:
        raise ValueError("Expected integer-like value, received None")
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to int") from exc


def load_materialized(path: str):
    """Load a dataset dict, rejecting streaming variants."""
    from datasets import (
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
        load_dataset,
    )

    dataset = load_dataset(path, streaming=False)
    if isinstance(dataset, DatasetDict):
        return dataset
    if isinstance(dataset, (IterableDatasetDict, IterableDataset)):
        raise TypeError(
            f"Dataset {path} loaded in streaming mode; disable streaming before preprocessing."
        )
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    raise TypeError(f"Unsupported dataset type {type(dataset)} for {path}")
