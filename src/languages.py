"""Get pronunciations from Wiktionary."""

from pathlib import Path

from iso639 import Language
import polars as pl
from rich.console import Console

from src import utils

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
            .then(pl.lit("Isolate"))
            .otherwise("family_name")
            .alias("family_id"),
        )
        .select(cols)
        .collect()
    )

    return pl.DataFrame(df)


def get_languages(
    file: Path | str = utils.OUT_PATH / "languages.parquet",
) -> pl.DataFrame:
    glottolog = get_glottolog()

    cols_select = [
        "iso639p3code",
        "lang",
        "glottocode",
        "family_id",
        "latitude",
        "longitude",
    ]

    df = glottolog.select(cols_select).sort("iso639p3code")
    df.write_parquet(Path(file))

    return df


def iso639_p3_to_p1(p3: str):
    try:
        lang = Language.from_part3(p3)
        return str(lang.part1)
    except (LookupError, ValueError, AttributeError):
        return


def iso639_p1_to_p3(p1: str):
    try:
        lang = Language.from_part1(p1)
        return str(lang.part3)
    except (LookupError, ValueError, AttributeError):
        return


def iso639_p3_to_name(p3: str):
    try:
        lang = Language.from_part3(p3)
        return str(lang.name)
    except (LookupError, ValueError, AttributeError):
        return


def iso639_name_to_p3(name: str):
    try:
        lang = Language.from_name(name)
        return str(lang.part3)
    except (LookupError, ValueError, AttributeError):
        return


def main():
    return get_languages()


if __name__ == "__main__":
    langs = main()
