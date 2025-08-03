"""Get pronunciations from Wiktionary."""

from pathlib import Path
from datetime import datetime
import polars as pl
import lingtypology
import wikipron
from rich.progress import track


def get_lang_populations(
    file: str | Path = Path("data", "koplenig2023.tsv")
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
        pl.read_csv(file, separator="\t")
        .rename(lambda x: x.lower())
        .select(["iso", "popsize"])
        .rename({"iso": "iso639p3code", "popsize": "pop_size"})
    )


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


def scrape_language(lang: str, conf: wikipron.Config):
    """Scrape pronunciations from language in Wiktionary.

    Args:
        lang (str): ISO-639-3 code of language.
        conf (wikipron.Config): _description_

    Raises:
        EmptyLanguage: _description_

    Returns:
        _type_: _description_
    """
    lang_name = lingtypology.glottolog.get_by_iso(lang)
    words, phons = [], []
    print(f"{lang_name} ({lang})...")
    for w, p in wikipron.scrape(conf):
        words.append(w)
        phons.append(p)
    df = pl.DataFrame({"word": words, "phon": phons})
    if len(df) == 0:
        raise EmptyLanguage()
    return df


def get_pronuciations(lang: str) -> pl.DataFrame:
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.

    Returns:
        pl.DataFrame: DataFrame with word and pronunciation pairs.
    """
    try:
        c = wikipron.Config(key=lang, syllable_boundaries=True, stress=True)
        df = scrape_language(lang, c)
        df.write_csv(Path("data", "raw") / f"{lang}.csv")
        return df
    except ValueError as e:
        print(e)
        return None
    except EmptyLanguage:
        print(f"{lang}: No broad pronunciations")
        try:
            c = wikipron.Config(key=lang, syllable_boundaries=True, stress=True)
            df = scrape_language(lang, c)
            df.write_csv(Path("data", "raw") / f"{lang}.csv")
            return df
        except EmptyLanguage:
            print(f"{lang}: No narrow pronunciations")
            return None
        except (ValueError, KeyError) as e:
            print(e)
            return None


def get_scraped_languages(
    path: Path = Path("data", "raw"), keep_empty: bool = False
) -> list[str]:
    """Evaluate is language has already been scraped.

    Args:
        path (str | Path): Path to raw data.

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
    n = len(langs)
    for i, la in track(enumerate(langs), f"Scraping {n} languages...", total=n):
        if la not in get_scraped_languages():
            print(f"{i + 1}) {datetime.now().strftime("%Y-%m-%d %H:%M")}")
            _ = get_pronuciations(la)
        else:
            print(f"{i + 1}) {la} has already been scraped")


def count_pronunciations(lang: str) -> int:
    """Count the number of scraped pronunciations in a langugae.

    Args:
        lang (str): ISO-639-3 code of language.

    Returns:
        int: Number of scraped pronunciations.
    """
    if lang not in get_scraped_languages():
        return 0
    return len(pl.read_csv(Path("data", "raw") / f"{lang}.csv"))


if __name__ == "__main__":
    scrape_all_languages()
