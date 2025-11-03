from pathlib import Path
from typing import Union, cast

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

DATASETS = {"xlwsd": "pasinit/xl-wsd", "xlwic": "pasinit/xl-wic", "mclwic": "mcl-wic/mcl_wic"}


def download_datasets(cache_root: Path = Path("data/raw")) -> None:
    """Download and persist all project datasets into data/raw/{alias}."""
    cache_root.mkdir(parents=True, exist_ok=True)
    for alias, hub_id in DATASETS.items():
        print(f"⏬ Downloading {alias} from {hub_id}")

        dataset = load_dataset(hub_id, streaming=False)
        if isinstance(dataset, (IterableDatasetDict, IterableDataset)):
            raise TypeError(f"Dataset {hub_id} loaded as streaming iterable; disable streaming before saving.")

        materialized = cast(Union[DatasetDict, Dataset], dataset)
        target_dir = cache_root / alias
        materialized.save_to_disk(target_dir)
        print(f"✅ saved to {target_dir}")


if __name__ == "__main__":
    download_datasets()
