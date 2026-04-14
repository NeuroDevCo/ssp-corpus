#!/usr/bin/env -S uv run --script

"""Get pronunciations from Wiktionary."""

from datetime import datetime
from logging import Logger, ERROR
from pathlib import Path

from languages import iso639_p3_to_name

from rich.console import Console
from wikipron import Config, scrape

from src.languages import get_languages

logger = Logger("pyppeteer", ERROR)

CON: Console = Console()


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


def scrape_language(
    lang: str,
    path: Path | str = Path("data", "raw"),
    delete_empty: bool = True,
    **kwargs,
):
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.
        path (Path, optional): Path to the folder in which the file will be created and saved: Should empty CSV be deleted? Defaults to True.
        delete_empty (bool, optional): Should empty CSV be deleted? Defaults to True.
        **kwargs: Additional arguments passed to `wikipron.Config`.
    """
    path = Path(path)
    time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    lang_label = f"{lang} ({iso639_p3_to_name(lang)})"
    CON.print(f"{lang_label} | [blue]Started[/blue] | {time}")

    kw = {"key": lang, "syllable_boundaries": True, "stress": True}
    for idx, mode in enumerate(["narrow", "broad"]):
        c = Config(**kw, narrow=mode == "narrow", **kwargs)

        fn = path / f"{lang}.csv"
        if fn.exists() and idx == 0:
            return

        with fn.open(encoding="utf-8", mode="a+") as f:
            i = 0
            for word, pron in scrape(c):
                f.write(word + "," + pron + "\n")
                print(word + " " + pron)
                i += 1

        if i == 0:
            CON.print(f"\t{mode}: {i} words found")

    if i == 0 and delete_empty:
        fn.unlink()


def scrape_all_languages(languages: str | list[str] | None = None, n_workers: int = 4):
    """Scrape all languages."""
    if isinstance(languages, str):
        languages = [languages]

    if languages is None:
        languages = get_languages().get_column("iso639p3code").to_list()

    for lang in languages:
        try:
            scrape_language(lang)
        except (ValueError, KeyError, KeyboardInterrupt, FileNotFoundError) as e:
            CON.print(f"\t[red]{e}[/red]")


if __name__ == "__main__":
    # parser = ArgumentParser(
    #     prog="Language scraper",
    #     description="Scrape a list of languages",
    #     epilog="Use Wikipron to scrape pronunciations from Wiktionary",
    # )
    # parser.add_argument("--language", "-l", default=None)
    # args = parser.parse_args()

    # scrape_all_languages(languages=args.language)

    scrape_all_languages(["aio", "fra"])
