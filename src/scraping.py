"""Get pronunciations from Wiktionary."""

from pathlib import Path
from functools import cache
from datetime import datetime
import polars as pl
import lingtypology
import wikipron
from rich.console import Console


CON = Console()
PHOIBLE = lingtypology.datasets.Phoible()
PHOIBLE.show_citation = False
GLOTTO = lingtypology.glottolog.glottolog


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


@cache
def get_lang_populations(
    file: str | Path = Path("data", "koplenig2019.csv"),
) -> pl.DataFrame:
    """Get Glottolog language populations from Ethnologue.

    Args:
        file (str, optional): Path to the TSV by Koplenig et al. (2023).

    Notes:
        Curated by Koplenig et al. (2023), rertieved from https://osf.io/p5ejc.

    References:

        Koplenig, A., Wolfer, S. & Meyer, P. A large quantitative analysis of written language challenges the idea that all languages are equally complex. Sci Rep 13, 15351 (2023). https://doi.org/10.1038/s41598-023-42327-3
    """
    return (
        pl.read_csv(file, separator=",")
        .rename(lambda x: x.lower())
        .select(["iso", "popsize"])
        .rename({"iso": "iso639p3code", "popsize": "pop_size"})
    )


@cache
def get_phoible() -> pl.DataFrame:
    """Get phonological repertoire size for each language."""
    names = ["glottocode", "phonemes", "consonants", "vowels"]
    phoible = PHOIBLE.get_df()[names]
    names = {"phonemes": "n_phon", "consonants": "n_cons", "vowels": "n_vow"}
    return pl.from_pandas(phoible).rename(names)


@cache
def get_glottolog() -> pl.DataFrame:
    """Get available languages in Glottolog.

    Returns:
        pl.DataFrame: Available languages in Glottolog.
    """
    return (
        pl.DataFrame(GLOTTO)
        .rename(lambda x: x.lower())
        .filter(pl.col("status") != "Extinct")
        .drop_nulls("iso639p3code")
        .with_columns(
            pl.when(pl.col("iso639p3code").is_in(["eus"]))
            .then(pl.lit("Isolate"))
            .otherwise("family_name")
            .alias("family_name")
        )
    )


@cache
def get_languages() -> pl.DataFrame:
    return (
        get_glottolog()
        .join(get_phoible(), on=["glottocode"], how="left")
        .join(get_lang_populations(), on=["iso639p3code"], how="left")
    )


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
    CON.print(f"\t[green]{lang}: {i} pronunciations found[/green]")
    return None


def scrape_language(lang: str, path: Path = Path("data", "raw")):
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.
        path (Path, optional): Path to the folder in which the file will be created and saved.
    """
    try:
        c = wikipron.Config(key=lang, syllable_boundaries=True, stress=True)
        _scrape_lang_wrapper(lang, c, path)
    except (ValueError, KeyError, FileNotFoundError) as e:
        CON.print(f"[red]{e}[/red]")
    except EmptyLanguage:
        CON.print(f"\t[yellow]{lang}: No broad pronunciations[/yellow]")
        try:
            c = wikipron.Config(key=lang, syllable_boundaries=True, stress=True)
            _scrape_lang_wrapper(lang, c, path)
        except EmptyLanguage:
            CON.print(f"\t[red]{lang}: No narrow pronunciations[/red]")
        except (ValueError, KeyError, FileNotFoundError) as e:
            CON.print(f"[red]{e}[/red]")


def get_scraped_languages(
    path: Path = Path("data", "raw"), keep_empty: bool = False
) -> list[str]:
    """Evaluate is language has already been scraped.

    Note:
        A language is considered to be scraped if a CSV file with the same ISO-639-3 code is present in the ``path`` folder.

    Args:
        path (Path, optional): Path to raw data. Defaults to ``Path("data", "raw")``.
        keep_empty (bool, optional): Should languages with no pronunciations available be included in the output? Defaults to False.

    Returns:
        list[str]: List of scraped languages.
    """
    if keep_empty:
        return [f.name.rsplit(".", 1)[0] for f in path.glob("*.csv")]
    files = path.glob("*.csv")
    return [f.name.rsplit(".", 1)[0] for f in files if f.stat().st_size > 1e3]


def scrape_all_languages():
    """Scrape all languages."""
    langs = get_languages().get_column("iso639p3code").to_list()
    langs = sorted([la for la in langs if la not in get_scraped_languages()])
    for i, la in enumerate(langs):
        time = datetime.now().strftime("%Y-%m-%d %H:%M")
        CON.print(f"[{i + 1}/{len(langs)}] {la} ({time})")
        scrape_language(la)


if __name__ == "__main__":
    scrape_all_languages()
