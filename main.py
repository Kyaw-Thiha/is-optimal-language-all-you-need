import typer
from InquirerPy import inquirer

from src.datahub import download_datasets, preprocess_datasets
from src.models import load_model, ModelKey
from experiments.ddi_xlwsd import run_ddi_xlwsd
from experiments.plots import plot_language_ddi, plot_language_traces

app = typer.Typer()


@app.command()
def dataset():
    download_datasets()
    preprocess_datasets()


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
