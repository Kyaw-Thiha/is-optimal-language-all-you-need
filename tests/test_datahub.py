"""Tests for the datahub loaders, preprocessors, and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import sys

import pytest
from datasets import Dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datahub.helpers import ensure_mapping, sample_to_record, safe_sequence, to_int
from src.datahub.loader import load_preprocessed
from src.datahub.preprocess import (
    preprocess_datasets,
    preprocess_mclwic,
    preprocess_xlwsd,
    preprocess_xlwic,
)
from src.datahub.sense_sample import SenseSample
from src.datahub.pipeline import DataRequest, prepare_datasets


# ---------------------------------------------------------------------------
# Helper fixtures and utilities


def _write_xlwsd_fixture(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    train_dir = raw_root / "xl-wsd" / "xl-wsd" / "training_datasets" / "wngt_examples_en"
    train_dir.mkdir(parents=True, exist_ok=True)

    xml_payload = """<corpus>
  <text id="d1" source="bank">
    <sentence id="d1.s1">
      <wf lemma="He" pos="PRON">He</wf>
      <wf lemma="go" pos="VERB">went</wf>
      <wf lemma="to" pos="ADP">to</wf>
      <wf lemma="the" pos="DET">the</wf>
      <instance id="d1.s1.t1" lemma="bank" pos="NOUN">bank</instance>
      <wf lemma="." pos="PUNCT">.</wf>
    </sentence>
  </text>
</corpus>
"""
    (train_dir / "wngt_examples_en.data.xml").write_text(xml_payload, encoding="utf-8")
    (train_dir / "wngt_examples_en.gold.key.txt").write_text("d1.s1.t1 bn:00031027n\n", encoding="utf-8")
    return raw_root


def _write_xlwic_fixture(tmp_path: Path) -> Path:
    """Create a minimal XL-WiC folder structure with French samples."""

    raw_root = tmp_path / "raw"
    fr_root = raw_root / "xlwic" / "xlwic_datasets" / "xlwic_wikt" / "french_fr"
    fr_root.mkdir(parents=True, exist_ok=True)

    validation_line = "\t".join(
        [
            "fleuve",
            "NOUN",
            "0",
            "1",
            "0",
            "1",
            "Le fleuve déborde.",
            "Le fleuve est paisible.",
            "1",
        ]
    )
    (fr_root / "fr_valid.txt").write_text(validation_line + "\n", encoding="utf-8")

    test_line = "\t".join(
        [
            "fleuve",
            "NOUN",
            "0",
            "1",
            "0",
            "1",
            "Il regarde la rive.",
            "La rive est boueuse.",
        ]
    )
    (fr_root / "fr_test_data.txt").write_text(test_line + "\n", encoding="utf-8")
    (fr_root / "fr_test_gold.txt").write_text("0\n", encoding="utf-8")
    return raw_root


def _write_mclwic_fixture(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    base = raw_root / "mcl-wic" / "all" / "MCL-WiC"
    for partition in ("training", "dev", "test"):
        part_dir = base / partition
        part_dir.mkdir(parents=True, exist_ok=True)
        split_name = {"training": "training", "dev": "dev", "test": "test"}[partition]
        data = [
            {
                "id": f"{split_name}.en-en.0",
                "lemma": "bank",
                "sentence1": "He went to the bank.",
                "sentence2": "She waited outside the bank.",
            }
        ]
        gold = [{"id": f"{split_name}.en-en.0", "tag": "T"}]
        (part_dir / f"{split_name}.en-en.data").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        (part_dir / f"{split_name}.en-en.gold").write_text(
            json.dumps(gold, ensure_ascii=False), encoding="utf-8"
        )
    return raw_root


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


# ---------------------------------------------------------------------------
# Preprocess tests


def test_preprocess_xlwsd(tmp_path: Path) -> None:
    raw_root = _write_xlwsd_fixture(tmp_path)

    preprocess_xlwsd(tmp_path, raw_root=raw_root)

    saved = load_from_disk(str(tmp_path / "xlwsd" / "train"))
    row = saved[0]

    assert row["dataset_id"] == "xlwsd"
    assert row["sense_tag"] == "bn:00031027n"
    assert row["target_span"] == [15, 19]


def test_preprocess_xlwic(tmp_path: Path) -> None:
    raw_root = _write_xlwic_fixture(tmp_path)

    preprocess_xlwic(tmp_path, raw_root=raw_root, configs=("fr",))

    validation = load_from_disk(str(tmp_path / "xlwic" / "validation"))
    row = validation[0]
    assert row["dataset_id"] == "xlwic"
    assert row["text_b"] == "Le fleuve est paisible."
    assert row["same_sense"] == 1

    test_split = load_from_disk(str(tmp_path / "xlwic" / "test"))
    assert test_split[0]["same_sense"] == 0


def test_preprocess_mclwic(tmp_path: Path) -> None:
    raw_root = _write_mclwic_fixture(tmp_path)

    preprocess_mclwic(tmp_path, raw_root=raw_root, splits=("all",))

    saved = load_from_disk(str(tmp_path / "mclwic" / "test"))
    row = saved[0]
    assert row["dataset_id"] == "mclwic"
    assert row["same_sense"] == 1


def test_preprocess_datasets_calls_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {"xlwsd": 0, "xlwic": 0, "mclwic": 0}

    monkeypatch.setattr(
        "src.datahub.preprocess.preprocess_xlwsd",
        lambda root, **kwargs: called.__setitem__("xlwsd", called["xlwsd"] + 1),
    )
    monkeypatch.setattr(
        "src.datahub.preprocess.preprocess_xlwic",
        lambda root, **kwargs: called.__setitem__("xlwic", called["xlwic"] + 1),
    )
    monkeypatch.setattr(
        "src.datahub.preprocess.preprocess_mclwic",
        lambda root, **kwargs: called.__setitem__("mclwic", called["mclwic"] + 1),
    )

    preprocess_datasets(tmp_path)

    assert called == {"xlwsd": 1, "xlwic": 1, "mclwic": 1}


def test_preprocess_datasets_subset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {"xlwsd": 0, "xlwic": 0, "mclwic": 0}

    monkeypatch.setattr(
        "src.datahub.preprocess.preprocess_xlwsd",
        lambda root, **kwargs: called.__setitem__("xlwsd", called["xlwsd"] + 1),
    )
    monkeypatch.setattr(
        "src.datahub.preprocess.preprocess_xlwic",
        lambda root, **kwargs: called.__setitem__("xlwic", called["xlwic"] + 1),
    )
    monkeypatch.setattr(
        "src.datahub.preprocess.preprocess_mclwic",
        lambda root, **kwargs: called.__setitem__("mclwic", called["mclwic"] + 1),
    )

    preprocess_datasets(tmp_path, datasets=("xlwsd", "mclwic", "xlwsd"))

    assert called == {"xlwsd": 1, "xlwic": 0, "mclwic": 1}


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
# Pipeline tests


def test_data_request_from_flags_all() -> None:
    request = DataRequest.from_flags(
        all=True,
        xl_wsd=False,
        xl_wic=False,
        mcl_wic=False,
        xlwic_config=["default", "de"],
        mclwic_splits=["trial"],
    )

    assert request.datasets == ("xlwsd", "xlwic", "mclwic")
    assert request.xlwic_configs == ("default", "de")
    assert request.mclwic_splits == ("trial",)


def test_data_request_requires_selection() -> None:
    with pytest.raises(ValueError):
        DataRequest.from_flags(
            all=False,
            xl_wsd=False,
            xl_wic=False,
            mcl_wic=False,
            xlwic_config=["default"],
            mclwic_splits=["all"],
        )


def test_prepare_datasets_invokes_selected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloads = []
    preprocess_calls = []

    monkeypatch.setattr(
        "src.datahub.pipeline.download_xlwic",
        lambda root, configs, force: downloads.append(("xlwic", tuple(configs), force, root)),
    )
    monkeypatch.setattr(
        "src.datahub.pipeline.download_mclwic",
        lambda root, splits, force: downloads.append(("mclwic", tuple(splits), force, root)),
    )
    monkeypatch.setattr(
        "src.datahub.pipeline.download_xlwsd",
        lambda root, force: downloads.append(("xlwsd", (), force, root)),
    )
    monkeypatch.setattr(
        "src.datahub.pipeline.preprocess_datasets",
        lambda output_root, **kwargs: preprocess_calls.append(
            {
                "output_root": output_root,
                "datasets": tuple(kwargs.get("datasets") or ()),
                "raw_root": kwargs.get("raw_root"),
                "xlwic_configs": tuple(kwargs.get("xlwic_configs") or ()),
                "mclwic_splits": tuple(kwargs.get("mclwic_splits") or ()),
            }
        ),
    )

    request = DataRequest(
        datasets=("xlwic", "mclwic"),
        xlwic_configs=("default", "de"),
        mclwic_splits=("trial",),
    )

    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "proc"

    prepare_datasets(request, raw_root=raw_root, processed_root=processed_root, force=True)

    assert downloads == [
        ("xlwic", ("default", "de"), True, raw_root),
        ("mclwic", ("trial",), True, raw_root),
    ]
    assert preprocess_calls == [
        {
            "output_root": processed_root,
            "datasets": ("xlwic", "mclwic"),
            "raw_root": raw_root,
            "xlwic_configs": ("default", "de"),
            "mclwic_splits": ("trial",),
        }
    ]


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
