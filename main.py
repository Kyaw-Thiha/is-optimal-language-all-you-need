import os
import typer
import numpy as np
from InquirerPy import inquirer

app = typer.Typer()


@app.command()
def main():
    print("Hey it ran!")


if __name__ == "__main__":
    app()
