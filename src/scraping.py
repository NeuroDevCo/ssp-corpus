"""Get pronunciations from Wiktionary."""

from functools import lru_cache
from pathlib import Path
from datetime import datetime

from lingtypology import datasets
import polars as pl
from rich.console import Console
from wikipron import Config, scrape

CON: Console = Console()


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


def get_glottolog() -> pl.DataFrame:
    url = "https://github.com/glottolog/glottolog-cldf/raw/refs/heads/master/cldf/languages.csv"
    cols = ["glottocode", "lang", "family_id", "iso639p3code", "latitude", "longitude"]

    df = (
        pl.scan_csv(url)
        .filter(pl.col("Level") == "language")
        .drop_nulls("ISO639P3code")
        .rename(lambda x: x.lower())
        .rename({"name": "lang"})
        .with_columns(
            pl.when(pl.col("is_isolate"))
            .then(pl.lit("Isolate"))
            .otherwise("family_id")
            .alias("family_id")
        )
        .select(cols)
    )

    return pl.DataFrame(df.collect())


def get_wikidata(file: Path | str = Path("data", "wikidata.csv")) -> pl.DataFrame:
    """
    Get number of speakers from Wikidata.

    Args:
        file (Path | str): Path to the CSV file to generate, if any. Defaults to data/wikidata.csv.

    Notes:
        sparQL query from https://gist.github.com/unhammer/3e8f2e0f79972bf5008a4c970081502d.
    """
    url = "https://gist.github.com/unhammer/3e8f2e0f79972bf5008a4c970081502d/raw/ffa2c42826bd2682f24781933e9a73a799c7f033/speakersPerLanguage.tsv"

    lookup = {
        "numberofspeakers": "n_speakers",
        "iso639_3": "iso639p3code",
        "speakerstime": "time",
        "item": "url",
    }
    df = (
        pl.scan_csv(url, separator="\t")
        .rename(lambda x: x.lower())
        .rename(lookup)
        .select(["iso639p3code", "n_speakers", "time", "url"])
    ).collect()

    df = pl.DataFrame(df)

    if file:
        df.write_csv(Path(file), separator=",")

    return df


@lru_cache
def get_phoible() -> pl.DataFrame:
    """Get phonological repertoire size for each language."""
    names = ["glottocode", "phonemes", "consonants", "vowels"]
    phoible = datasets.Phoible().get_df()[names]
    names = {"phonemes": "n_phon", "consonants": "n_cons", "vowels": "n_vow"}

    return pl.from_pandas(phoible).rename(names)


@lru_cache
def get_languages(file: Path | str = Path("out", "languages.csv")) -> pl.DataFrame:
    glottolog = get_glottolog()
    phoible = get_phoible()
    wikidata_path = Path("data/wikidata.csv")
    wikidata = pl.read_csv(wikidata_path) if wikidata_path.exists() else get_wikidata()
    wikidata = wikidata.select(["iso639p3code", "n_speakers"])

    cols_select = [
        "iso639p3code",
        "lang",
        "glottocode",
        "n_speakers",
        "family_id",
        "latitude",
        "longitude",
        "n_phon",
        "n_cons",
        "n_vow",
    ]
    df = (
        glottolog.join(wikidata, on=["iso639p3code"], how="left")
        .join(phoible, on=["glottocode"], how="left")
        .select(cols_select)
        .sort("iso639p3code")
    )

    file = Path(file)
    if file:
        df.write_csv(file)

    return df


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
