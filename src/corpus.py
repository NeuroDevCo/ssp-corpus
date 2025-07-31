"""Preprocess CHILDES corpora."""

from os import walk
from pathlib import Path
from collections import Counter
import pandas as pd
from rich.progress import track
from rich import console
import pylangacq

con = console.Console()


def find_datasets(corpus: str) -> list[Path]:
    """Find datasets in corpus.

    Args:
        corpus (str): _description_

    Returns:
        list[Path]: _description_
    """
    datasets = []
    for root, _, files in walk(corpus):
        for f in files:
            datasets.append(Path(root, f))
    return [d for d in datasets if str(d).endswith(".cha")]


def process_corpus(path: str) -> pd.DataFrame:
    """Make DataFrame with word frequencies.

    Args:
        freqs (Counter): Word frequencies.
        file (str): Path to the output CSV file.

    Returns:
        pd.DataFrame: _description_
    """
    corpora = [Path(root, d) for root, dirs, _ in walk(path) for d in dirs]
    con.rule(f"Processing language [bold]{path.name}[/bold]")
    for corpus in track(corpora):
        datasets = find_datasets(corpus)
        con.print(f"Corpus {corpus.name} ({len(datasets)} datasets)")
        table = pd.DataFrame(columns=["corpus", "subdir", "dataset", "token", "count"])
        errors = []
        for dataset in enumerate(datasets):
            tokens = []
            try:
                data = pylangacq.read_chat(str(dataset))
                tokens.extend(data.words())
                words, counts = [], []
                for word, count in Counter(tokens).items():
                    words.append(word)
                    counts.append(count)
                data = {"corpus": corpus.name, "word": words, "count": counts}
                table = pd.concat([table, pd.DataFrame(data)])
            except ValueError:
                errors.append(dataset.name)
    con.print(f"[red]Error reading {len(errors)} datasets: {errors}[/red]")
    table = table.groupby(["corpus", "token"]).sum()
    table = table.sort_values(by="count", ascending=False).reset_index(drop=True)
    return table


if __name__ == "__main__":
    COUNTS = process_corpus(Path("data", "CHILDES", "ENG-NA"))
