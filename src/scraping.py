"""Get pronunciations from Wiktionary."""

from datetime import datetime
from pathlib import Path

from rich.console import Console
from wikipron import Config, scrape

from src.languages import get_languages

CON: Console = Console()


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


def _scrape_lang_wrapper(lang: str, conf: Config, path: Path = Path("data", "raw")):
    """Wrapper for scraping pronunciations from language in Wiktionary.

    Args:
        lang (str): ISO-639-3 code of language.
        conf (Config): Scraping params.

    Raises:
        EmptyLanguage: No available pronunciations for language.
    """
    fn = path / f"{lang}.csv"
    if fn.exists():
        return

    i = 0
    with fn.open(encoding="utf-8", mode="a+") as f:
        for w, p in scrape(conf):
            f.writelines(w + "," + p + "\n")
            i += 1

    if i == 0:
        raise EmptyLanguage()

    CON.print(f"\t[green]{lang}: {i} pronunciations found[/green]")


def scrape_language(lang: str, path: Path | str = Path("data", "raw")):
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.
        path (Path, optional): Path to the folder in which the file will be created and saved.
    """
    for mode in ["broad", "narrow"]:
        is_narrow = mode == "narrow"
        c = Config(key=lang, syllable_boundaries=True, stress=True, narrow=is_narrow)

        try:
            _scrape_lang_wrapper(lang, c, Path(path))
            return
        except (ValueError, KeyError, FileNotFoundError) as e:
            CON.print(f"[red]{e}[/red]")
        except EmptyLanguage:
            CON.print(f"\t[yellow]{lang}: No broad pronunciations[/yellow]")
    else:
        CON.print(f"\t[red]{lang}: No narrow pronunciations[/red]")


def get_scraped_languages(
    path: Path | str = Path("data", "raw"), keep_empty: bool = False
) -> list[str]:
    """Evaluate is language has already been scraped.

    Note:
        A language is considered to be scraped if a CSV file with the same ISO-639-3 code is present in the ``path`` folder.

    Args:
        path (Path | str, optional): Path to raw data. Defaults to ``Path("data", "raw")``.
        keep_empty (bool, optional): Should languages with no pronunciations available be included in the output? Defaults to False.

    Returns:
        list[str]: List of scraped languages.
    """
    files = Path(path).glob("*.csv")

    if keep_empty:
        return [f.stem for f in files]

    return [f.stem for f in files if f.stat().st_size > 1e3]


def scrape_all_languages():
    """Scrape all languages."""
    langs = get_languages().get_column("iso639p3code").to_list()
    langs = sorted(la for la in langs if la not in get_scraped_languages())

    for i, la in enumerate(langs):
        time = datetime.now().isoformat()
        CON.print(f"[{i + 1}/{len(langs)}] {la} ({time})")
        scrape_language(la)


if __name__ == "__main__":
    scrape_all_languages()
