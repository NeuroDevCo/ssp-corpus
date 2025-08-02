"""Get pronunciations from Wiktionary."""

from pathlib import Path
from datetime import datetime
import polars as pl
import lingtypology
import wikipron
from rich.progress import track


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


def get_scraped_languages(path: Path = Path("data", "raw")) -> list[str]:
    """Evaluate is language has already been scraped.

    Args:
        path (str | Path): Path to raw data.

    Returns:
        list[str]: List of scraped languages.
    """
    return [f.name.rsplit(".", 1)[0] for f in path.glob("*.csv")]


def scrape_all_languages():
    """Scrape all languages."""
    langs = get_languages().get_column("iso639p3code").to_list()
    n = len(langs)
    for i, l in track(enumerate(langs), f"Scraping {n} languages...", total=n):
        if l not in get_scraped_languages():
            print(f"{i + 1}) {datetime.now().strftime("%Y-%m-%d %H:%M")}")
            _ = get_pronuciations(l)
        else:
            print(f"{i + 1}) {l} has already been scraped")


if __name__ == "__main__":
    scrape_all_languages()
