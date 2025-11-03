"""Tests for the datahub loaders, preprocessors, and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import sys

import pytest
import datasets as hf_datasets
from datasets import Dataset, DatasetDict, IterableDataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datahub.download import download_datasets
from src.datahub.helpers import (
    ensure_mapping,
    load_materialized,
    sample_to_record,
    safe_sequence,
    to_int,
)
from src.datahub.loader import load_preprocessed
from src.datahub.preprocess import (
    preprocess_datasets,
    preprocess_mclwic,
    preprocess_xlwsd,
    preprocess_xlwic,
)
from src.datahub.sense_sample import SenseSample


# ---------------------------------------------------------------------------
# Helper fixtures and utilities


def _make_dataset_dict(rows_per_split: Mapping[str, Iterable[Mapping[str, object]]]) -> DatasetDict:
    """Utility to build a DatasetDict with provided rows."""

    return DatasetDict(
        {split: Dataset.from_list(list(rows)) for split, rows in rows_per_split.items()}
    )


# ---------------------------------------------------------------------------
# Helper utility tests


def test_sample_to_record_converts_tuple_span() -> None:
    sample = SenseSample(
        sample_id="id",
        dataset_id="ds",
        split="train",
        language="en",
        text_a="Hello",
        text_b=None,
        lemma="hello",
        target_span=(1, 4),
        sense_tag=None,
        same_sense=None,
    )
    record = sample_to_record(sample)
    assert record["target_span"] == [1, 4]


def test_safe_sequence_handles_various_inputs() -> None:
    assert safe_sequence(None) == []
    assert safe_sequence(["a", "b"]) == ["a", "b"]
    assert safe_sequence("token") == ["token"]


def test_to_int_accepts_multiple_types() -> None:
    assert to_int(3) == 3
    assert to_int(True) == 1
    assert to_int("7") == 7
    with pytest.raises(ValueError):
        to_int(None)


def test_ensure_mapping_rejects_invalid() -> None:
    assert ensure_mapping({"a": 1}) == {"a": 1}
    with pytest.raises(TypeError):
        ensure_mapping(42)


def test_load_materialized_dataset_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dict = _make_dataset_dict(
        {"train": [{"language": "en", "context": "hello", "lemma": "hi"}]}
    )

    monkeypatch.setattr(
        hf_datasets,
        "load_dataset",
        lambda path, streaming=False: dataset_dict,
    )

    materialized = load_materialized("dummy/path")
    assert isinstance(materialized, DatasetDict)
    assert "train" in materialized


def test_load_materialized_wraps_single_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset.from_list([{"language": "en", "context": "hi", "lemma": "hello"}])

    monkeypatch.setattr(
        hf_datasets,
        "load_dataset",
        lambda path, streaming=False: dataset,
    )

    materialized = load_materialized("single/dataset")
    assert isinstance(materialized, DatasetDict)
    assert "train" in materialized


def test_load_materialized_rejects_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    streaming_ds = IterableDataset.from_generator(lambda: iter([{"a": 1}]))

    monkeypatch.setattr(
        hf_datasets,
        "load_dataset",
        lambda path, streaming=False: streaming_ds,
    )

    with pytest.raises(TypeError):
        load_materialized("streaming/dataset")


# ---------------------------------------------------------------------------
# Preprocess tests


def test_preprocess_xlwsd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_dataset_dict(
        {
            "train": [
                {
                    "language": "en",
                    "context": "He went to the bank.",
                    "lemma": "bank",
                    "target_start_char": 3,
                    "target_end_char": 7,
                    "sense_keys": ["bank%1:14:01::"],
                }
            ]
        }
    )

    monkeypatch.setattr("src.datahub.preprocess.load_materialized", lambda _: raw)

    preprocess_xlwsd(tmp_path)

    saved = load_from_disk(str(tmp_path / "xlwsd" / "train"))
    row = saved[0]

    assert row["dataset_id"] == "xlwsd"
    assert row["sense_tag"] == "bank%1:14:01::"
    assert row["target_span"] == [3, 7]


def test_preprocess_xlwic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_dataset_dict(
        {
            "validation": [
                {
                    "language": "fr",
                    "sentence1": "Le fleuve déborde.",
                    "sentence2": "Le fleuve est paisible.",
                    "lemma": "fleuve",
                    "label": 1,
                }
            ]
        }
    )

    monkeypatch.setattr("src.datahub.preprocess.load_materialized", lambda _: raw)

    preprocess_xlwic(tmp_path)

    saved = load_from_disk(str(tmp_path / "xlwic" / "validation"))
    row = saved[0]

    assert row["dataset_id"] == "xlwic"
    assert row["text_b"] == "Le fleuve est paisible."
    assert row["same_sense"] == 1


def test_preprocess_mclwic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_dataset_dict(
        {
            "test": [
                {
                    "language": "tr",
                    "sentence1": "Çocuk ağaçtan düştü.",
                    "sentence2": "Değerler düştü.",
                    "lemma": "düşmek",
                    "label": 0,
                }
            ]
        }
    )

    monkeypatch.setattr("src.datahub.preprocess.load_materialized", lambda _: raw)

    preprocess_mclwic(tmp_path)

    saved = load_from_disk(str(tmp_path / "mclwic" / "test"))
    row = saved[0]
    assert row["dataset_id"] == "mclwic"
    assert row["same_sense"] == 0


def test_preprocess_datasets_calls_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {"xlwsd": 0, "xlwic": 0, "mclwic": 0}

    monkeypatch.setattr("src.datahub.preprocess.preprocess_xlwsd", lambda root: called.__setitem__("xlwsd", called["xlwsd"] + 1))
    monkeypatch.setattr("src.datahub.preprocess.preprocess_xlwic", lambda root: called.__setitem__("xlwic", called["xlwic"] + 1))
    monkeypatch.setattr("src.datahub.preprocess.preprocess_mclwic", lambda root: called.__setitem__("mclwic", called["mclwic"] + 1))

    preprocess_datasets(tmp_path)

    assert called == {"xlwsd": 1, "xlwic": 1, "mclwic": 1}


# ---------------------------------------------------------------------------
# Loader tests


def test_load_preprocessed_round_trip(tmp_path: Path) -> None:
    dataset_root = tmp_path / "cache"
    (dataset_root / "xlwic").mkdir(parents=True, exist_ok=True)
    records = [
        {
            "sample_id": "xlwic-train-0",
            "dataset_id": "xlwic",
            "split": "train",
            "language": "en",
            "text_a": "The bank is crowded.",
            "text_b": "She went to the bank.",
            "lemma": "bank",
            "target_span": None,
            "sense_tag": None,
            "same_sense": 1,
        }
    ]
    Dataset.from_list(records).save_to_disk(dataset_root / "xlwic" / "train")

    samples = list(load_preprocessed("xlwic", "train", root=dataset_root))
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, SenseSample)
    assert sample.same_sense == 1
    assert sample.text_b == "She went to the bank."


def test_load_preprocessed_span_conversion(tmp_path: Path) -> None:
    dataset_root = tmp_path / "cache"
    (dataset_root / "xlwsd").mkdir(parents=True, exist_ok=True)

    records = [
        {
            "sample_id": "xlwsd-test-0",
            "dataset_id": "xlwsd",
            "split": "test",
            "language": "en",
            "text_a": "He fell by the river bank.",
            "text_b": None,
            "lemma": "bank",
            "target_span": [3, 7],
            "sense_tag": "bank%1:14:01::",
            "same_sense": None,
        }
    ]
    Dataset.from_list(records).save_to_disk(dataset_root / "xlwsd" / "test")

    sample = next(load_preprocessed("xlwsd", "test", root=dataset_root))
    assert sample.target_span == (3, 7)
    assert sample.sense_tag == "bank%1:14:01::"


# ---------------------------------------------------------------------------
# Download utility tests


def test_download_datasets_saves_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(hub_id: str, streaming: bool = False):
        return DatasetDict({"train": Dataset.from_list([{"hub": hub_id}])})

    monkeypatch.setattr("src.datahub.download.load_dataset", fake_load_dataset)

    download_datasets(cache_root=tmp_path)

    for alias in ("xlwsd", "xlwic", "mclwic"):
        assert (tmp_path / alias).exists()


def test_download_datasets_rejects_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    streaming_ds = IterableDataset.from_generator(lambda: iter([{"a": 1}]))

    monkeypatch.setattr("src.datahub.download.load_dataset", lambda *args, **kwargs: streaming_ds)

    with pytest.raises(TypeError):
        download_datasets(cache_root=tmp_path)


# ---------------------------------------------------------------------------
# Invariants / property-style checks


def test_sense_sample_invariants(tmp_path: Path) -> None:
    dataset_root = tmp_path / "store"
    (dataset_root / "mclwic").mkdir(parents=True, exist_ok=True)
    records = [
        {
            "sample_id": "mclwic-train-0",
            "dataset_id": "mclwic",
            "split": "train",
            "language": "de",
            "text_a": "Er fiel vom Baum.",
            "text_b": "Die Aktien fielen schnell.",
            "lemma": "fallen",
            "target_span": None,
            "sense_tag": None,
            "same_sense": 0,
        }
    ]
    Dataset.from_list(records).save_to_disk(dataset_root / "mclwic" / "train")

    sample = next(load_preprocessed("mclwic", "train", root=dataset_root))
    assert sample.same_sense in (0, 1, None)
    assert sample.target_span in (None, (0, 0)) or isinstance(sample.target_span, tuple)
