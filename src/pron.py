"""Import pronunciations from Wiktionary."""

from pathlib import Path
import polars as pl
from iso639.language import ALL_LANGUAGES
import wikipron as wkp
from pyglottolog import Glottolog
import lingtypology


def get_iso_langs() -> dict[str, str]:
    """Get the name and code of all languages in ISO-639-3.

    Returns:
        dict[str, str]: Code and name of each language.
    """
    return {la.part3: la.name for la in ALL_LANGUAGES}


def scrape_lang(lang: str) -> pl.DataFrame:
    if lang not in get_iso_langs():
        raise ValueError("`lang` must be a valid code in ISO-639")
    config = wkp.Config(key=lang, syllable_boundaries=True, narrow=True)
    return config
    # for word, pron in wkp.scrape(config):


scrape_lang("ara")
