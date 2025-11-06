import typer
from InquirerPy import inquirer

from src.datahub import download_datasets, preprocess_datasets
from src.models import load_model
from experiments.ddi import run_ddi_xlwsd

app = typer.Typer()


@app.command()
def dataset():
    download_datasets()
    preprocess_datasets()


@app.command()
def ddi_xlwsd():
    run_ddi_xlwsd("minilm")


@app.command()
def infer(model: str = "", text: str = ""):
    runner = load_model("llama3", device="cuda:0")
    batch = runner.tokenize(["Merhaba dünya"])
    outputs = runner.forward(batch)

    return outputs


if __name__ == "__main__":
    app()
