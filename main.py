import os
import typer
import numpy as np
from InquirerPy import inquirer

from src.datahub import download_datasets, preprocess_datasets

app = typer.Typer()


@app.command()
def dataset():
    download_datasets()
    preprocess_datasets()


if __name__ == "__main__":
    app()
