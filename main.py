from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from InquirerPy import inquirer

from experiments.ddi_xlwsd import run_ddi_xlwsd
from experiments.plots import PlotSaveConfig, plot_language_ddi, plot_language_traces
from src.datahub import DataRequest, prepare_datasets
from src.models import ModelKey, load_model

app = typer.Typer()


@app.command("datahub")
def datahub(
    all: bool = typer.Option(False, "--all", help="Download every dataset."),
    xl_wsd: bool = typer.Option(False, "--xl-wsd", help="Download XL-WSD."),
    xl_wic: bool = typer.Option(False, "--xl-wic", help="Download XL-WiC."),
    mcl_wic: bool = typer.Option(False, "--mcl-wic", help="Download MCL-WiC."),
    xlwic_config: List[str] = typer.Option(
        ["default"],
        "--xlwic-config",
        help="XL-WiC configs to fetch (matching pasinit/xlwic).",
        show_default=True,
    ),
    mclwic_splits: List[str] = typer.Option(
        ["all"],
        "--mclwic-splits",
        help="MCL-WiC bundles to pull (all, test-gold, trial).",
        show_default=True,
    ),
    force: bool = typer.Option(False, "--force", help="Redownload even if files exist."),
    raw_root: Path = typer.Option(
        Path("data/raw"),
        "--raw-root",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory to store raw corpora.",
    ),
    processed_root: Path = typer.Option(
        Path("data/preprocess"),
        "--processed-root",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory to store SenseSample caches.",
    ),
) -> None:
    """
    Download requested datasets and materialize them into the SenseSample schema.
    """
    try:
        request = DataRequest.from_flags(
            all=all,
            xl_wsd=xl_wsd,
            xl_wic=xl_wic,
            mcl_wic=mcl_wic,
            xlwic_config=xlwic_config,
            mclwic_splits=mclwic_splits,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    prepare_datasets(
        request,
        raw_root=raw_root,
        processed_root=processed_root,
        force=force,
    )


@app.command()
def ddi_xlwsd(
    model_name: ModelKey = typer.Option("minilm", "--model", help="Model key to evaluate."),
    probe_name: str = typer.Option(
        "linear-logistic",
        "--probe",
        help="Probe to use (e.g. linear-logistic, random-forest, mlp).",
    ),
    tune_probe: bool = typer.Option(
        False,
        "--tune-probe",
        help="Enable Optuna tuning (currently only supported for random-forest).",
    ),
    tuning_trials: int = typer.Option(
        25,
        "--tuning-trials",
        help="Number of Optuna trials when --tune-probe is set.",
    ),
    plots_root: Optional[Path] = typer.Option(
        None,
        "--plots-root",
        help="Directory where plots should be saved (subfolders are created automatically).",
    ),
    plots_tag: Optional[str] = typer.Option(
        None,
        "--plots-tag",
        help="Folder suffix for this run (defaults to timestamp).",
    ),
    save_static: bool = typer.Option(True, help="Write static PNG snapshots when saving plots."),
    save_html: bool = typer.Option(True, help="Write interactive HTML plots when saving."),
    batch_size: int = typer.Option(256, "--batch-size", help="Batch size for model forward passes."),
):
    summaries, lemma_traces, records = run_ddi_xlwsd(
        model_name,
        batch_size=batch_size,
        probe_name=probe_name,
        tune_probe=tune_probe,
        tuning_trials=tuning_trials,
    )

    save_config: Optional[PlotSaveConfig] = None
    if plots_root:
        tag = plots_tag or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base_dir = plots_root / "ddi_xlwsd" / model_name
        save_config = PlotSaveConfig(base_dir=base_dir, run_tag=tag, save_static=save_static, save_html=save_html)
        print(f"[plots] Saving figures under {base_dir / tag}")

    plot_language_ddi(
        summaries,
        model_name,
        save_to=save_config.for_plot("language_ddi") if save_config else None,
    )
    plot_language_traces(
        lemma_traces,
        model_name,
        save_to=save_config.for_plot("language_traces") if save_config else None,
    )


@app.command()
def infer(model: str = "", text: str = ""):
    runner = load_model("llama3", device="cuda:0")
    batch = runner.tokenize(["Merhaba dünya"])
    outputs = runner.forward(batch)

    return outputs


if __name__ == "__main__":
    app()
