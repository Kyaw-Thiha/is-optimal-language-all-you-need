from pathlib import Path

from .helpers import load_materialized

DATASETS = {"xlwsd": "pasinit/xl-wsd", "xlwic": "pasinit/xl-wic", "mclwic": "mcl-wic/mcl_wic"}


def download_datasets(cache_root: Path = Path("data/raw")) -> None:
    """Download and persist all project datasets into data/raw/{alias}."""
    cache_root.mkdir(parents=True, exist_ok=True)
    for alias, hub_id in DATASETS.items():
        print(f"⏬ Downloading {alias} from {hub_id}")

        materialized = load_materialized(hub_id)
        target_dir = cache_root / alias
        materialized.save_to_disk(target_dir)
        print(f"✅ saved to {target_dir}")


if __name__ == "__main__":
    download_datasets()
