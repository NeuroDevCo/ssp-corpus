#!/usr/bin/env -S uv run --script

"""Get pronunciations from Wiktionary."""

from argparse import ArgumentParser
from datetime import datetime
from logging import Logger, ERROR
from pathlib import Path


from rich.console import Console
from wikipron import Config, scrape

from src.languages import get_languages

logger = Logger("pyppeteer", ERROR)

CON: Console = Console()


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


def get_scraped_languages(
    path: Path | str = Path("data", "raw"),
    keep_empty: bool = False,
    min_size: int = 1_000,
    extension: str = ".csv",
) -> list[str]:
    """Evaluate is language has already been scraped.

    Note:
        A language is considered to be scraped if a CSV file with the same ISO-639-3 code is present in the ``path`` folder.

    Args:
        path (Path | str, optional): Path to raw data. Defaults to ``Path("data", "raw")``.
        keep_empty (bool, optional): Should languages with no pronunciations available be included in the output? Defaults to False.
        min_size (int, optional): Number of minimum bits for a file to be considered non-empty. Defaults to 1_000.
        extension (str, optional): File extension to look for. Defaults to ".csv".

    Returns:
        list[str]: List of scraped languages.
    """
    files = Path(path).glob("*" + extension)

    if keep_empty:
        return [f.stem for f in files]

    return [f.stem for f in files if f.stat().st_size > min_size]


def scrape_language(lang: str, path: Path | str = Path("data", "raw")):
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.
        path (Path, optional): Path to the folder in which the file will be created and saved.
    """
    path = Path(path)
    for mode_idx, mode in enumerate(["narrow", "broad"]):
        delete_empty = False

        try:
            c = Config(
                key=lang, syllable_boundaries=True, stress=True, narrow=mode == "narrow"
            )
            time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

            if mode_idx == 0:
                CON.print(f"{lang} | [blue]Started[/blue] | {time}")
                delete_empty = True

            fn = path / f"{lang}.csv"
            if fn.exists():
                return

            i = 0
            with fn.open(encoding="utf-8", mode="a+") as f:
                for w, p in scrape(c):
                    f.write(w + "," + p + "\n")
                    i += 1

            if i == 0:
                if delete_empty:
                    fn.unlink()
                raise EmptyLanguage()

            CON.print(f"{lang} | [green]Finished ({i} found) [/green] | {time}")
            return
        except (ValueError, KeyError, FileNotFoundError) as e:
            CON.print(f"[red]{e}[/red]")
        except EmptyLanguage:
            if mode_idx == 0:
                CON.print(f"\t[yellow]{lang}: No {mode} words[/yellow]")
            pass
    CON.print(f"\t[red]{lang}: No {mode} words either[/red]")
    time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    CON.print(f"{lang} | [red]Finished (no words found)[/red] | {time}")


def scrape_all_languages(languages: str | list[str] | None = None, n_workers: int = 4):
    """Scrape all languages."""
    if isinstance(languages, str):
        languages = [languages]

    if languages is None:
        languages = get_languages().get_column("iso639p3code").to_list()

    languages = sorted(la for la in languages if la not in get_scraped_languages())

    for lang in languages:
        scrape_language(lang)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Language scraper",
        description="Scrape a list of languages",
        epilog="Use Wikipron to scrape pronunciations from Wiktionary",
    )
    parser.add_argument("--language", "-l", default=None)
    args = parser.parse_args()

    scrape_all_languages(languages=args.language)
