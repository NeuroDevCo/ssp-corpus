"""Get pronunciations from Wiktionary."""

from functools import lru_cache
from pathlib import Path

from lingtypology import datasets
import polars as pl
from rich.console import Console

CON: Console = Console()


class EmptyLanguage(Exception):
    "Language has no available pronunciations."


def get_glottolog() -> pl.DataFrame:
    url = "https://github.com/glottolog/glottolog-cldf/raw/refs/heads/master/cldf/"

    cols = [
        "glottocode",
        "lang",
        "family_id",
        "family_name",
        "iso639p3code",
        "latitude",
        "longitude",
    ]

    rm_str = "Glottolog classification of the languages in the family "
    df_trees = (
        pl.scan_csv(url + "trees.csv")
        .with_columns(pl.col("Description").str.replace(rm_str, ""))
        .rename({"ID": "Family_ID", "Description": "Family_Name"})
    )
    df = (
        pl.scan_csv(url + "languages.csv")
        .join(df_trees, on="Family_ID", how="full")
        .filter(pl.col("Level") == "language")
        .drop_nulls("ISO639P3code")
        .rename(lambda x: x.lower())
        .rename({"name": "lang"})
        .with_columns(
            pl.when(pl.col("is_isolate"))
            .then(pl.lit("isolate"))
            .otherwise("family_name")
            .alias("family_id"),
            pl.when(pl.col("is_isolate"))
            .then(pl.lit("Isolate"))
            .otherwise("family_name")
            .alias("family_name"),
        )
        .select(cols)
        .collect()
    )

    return pl.DataFrame(df)


@lru_cache
def get_phoible() -> pl.DataFrame:
    """Get phonological repertoire size for each language."""
    names = ["glottocode", "phonemes", "consonants", "vowels"]
    phoible = datasets.Phoible()
    phoible.show_citation = False
    phoible = pl.from_pandas(phoible.get_df()[names]).unique("glottocode")
    names = {
        "phonemes": "phon_size_all",
        "consonants": "phon_size_c",
        "vowels": "phon_size_v",
    }

    return phoible.rename(names)


def get_languages(file: Path | str = Path("out", "languages.parquet")) -> pl.DataFrame:
    glottolog = get_glottolog()
    phoible = get_phoible()

    cols_select = [
        "iso639p3code",
        "lang",
        "glottocode",
        "family_id",
        "latitude",
        "longitude",
        "phon_size_all",
        "phon_size_c",
        "phon_size_v",
    ]
    df = (
        glottolog.join(phoible, on=["glottocode"], how="left")
        .select(cols_select)
        .sort("iso639p3code")
    )

    if file:
        df.write_parquet(Path(file))

    return df


if __name__ == "__main__":
    LANG = get_languages()
