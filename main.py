from pathlib import Path
from typing import List

import typer
from InquirerPy import inquirer

from experiments.ddi_xlwsd import run_ddi_xlwsd
from experiments.plots import plot_language_ddi, plot_language_traces
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


# Backward-compatible alias: `python main.py dataset ...`
app.command("dataset")(datahub)


@app.command()
def ddi_xlwsd():
    model_name: ModelKey = "minilm"
    summaries, lemma_traces, records = run_ddi_xlwsd(model_name)

    # Plotting summary of all languages
    plot_language_ddi(summaries, model_name)

    # Plotting per language traces
    plot_language_traces(lemma_traces, model_name)


@app.command()
def infer(model: str = "", text: str = ""):
    runner = load_model("llama3", device="cuda:0")
    batch = runner.tokenize(["Merhaba dünya"])
    outputs = runner.forward(batch)

    return outputs


if __name__ == "__main__":
    app()
