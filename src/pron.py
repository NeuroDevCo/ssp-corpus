"""Get pronunciations from Wiktionary."""

from pathlib import Path
from datetime import datetime
import polars as pl
import lingtypology
import wikipron
from rich.progress import track
from rich.console import Console

con = Console()


def get_languages() -> pl.DataFrame:
    """Get available languages in Glottolog.

    Returns:
        pl.DataFrame: Available languages in Glottolog.
    """
    return (
        pl.DataFrame(lingtypology.glottolog.glottolog)
        .rename(lambda x: x.lower())
        .filter(pl.col("status") != "Extinct")
        .drop_nulls("iso639p3code")
        .with_columns(
            pl.when(pl.col("iso639p3code").is_in(["eus"]))
            .then(pl.lit("Isolate"))
            .otherwise("family_name")
            .alias("family_name")
        )
        .join(get_lang_populations(), on=["iso639p3code"])
    )


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


def _scrape_lang_wrapper(
    lang: str, conf: wikipron.Config, path: Path = Path("data", "raw")
):
    """Wrapper for scraping pronunciations from language in Wiktionary.

    Args:
        lang (str): ISO-639-3 code of language.
        conf (wikipron.Config): Scraping params.

    Raises:
        EmptyLanguage: No available pronunciations for language.
    """
    fname = path / f"{lang}.csv"
    if fname.exists():
        return None
    i = 0
    with open(fname, encoding="utf-8", mode="a+") as f:
        for w, p in wikipron.scrape(conf):
            f.writelines(w + "," + p + "\n")
            i += 1
    if i == 0:
        raise EmptyLanguage()
    con.print(f"\t[green]{lang}: {i} pronunciations found[/green]")
    return None


def scrape_lang(lang: str, path: Path = Path("data", "raw")):
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.
        path (Path, optional): Path to the folder in which the file will be created and saved.
    """
    try:
        c = wikipron.Config(key=lang, syllable_boundaries=True, stress=True)
        _scrape_lang_wrapper(lang, c, path)
    except (ValueError, KeyError, FileNotFoundError) as e:
        con.print(f"[red]{e}[/red]")
    except EmptyLanguage:
        con.print(f"\t[yellow]{lang}: No broad pronunciations[/yellow]")
        try:
            c = wikipron.Config(key=lang, syllable_boundaries=True, stress=True)
            _scrape_lang_wrapper(lang, c, path)
        except EmptyLanguage:
            con.print(f"\t[red]{lang}: No narrow pronunciations[/red]")
        except (ValueError, KeyError, FileNotFoundError) as e:
            con.print(f"[red]{e}[/red]")


def get_scraped_languages(
    path: Path = Path("data", "raw"), keep_empty: bool = False
) -> list[str]:
    """Evaluate is language has already been scraped.

    Args:
        path (Path): Path to raw data.

    Returns:
        list[str]: List of scraped languages.
    """
    if keep_empty:
        return [f.name.rsplit(".", 1)[0] for f in path.glob("*.csv")]
    return [
        f.name.rsplit(".", 1)[0] for f in path.glob("*.csv") if f.stat().st_size > 1e3
    ]


def scrape_all_languages():
    """Scrape all languages."""
    langs = get_languages().get_column("iso639p3code").to_list()
    langs = sorted([l for l in langs if l not in get_scraped_languages()])
    for i, l in enumerate(langs):
        time = datetime.now().strftime("%Y-%m-%d %H:%M")
        con.print(f"[{i + 1}/{len(langs)}] {l} ({time})")
        scrape_lang(l)


if __name__ == "__main__":
    scrape_all_languages()
