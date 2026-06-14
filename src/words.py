from pathlib import Path

import polars as pl
from rich.progress import track

from src import languages, utils


def get_lang_words(
    file: Path | str, syll_boundaries: list[str] | None = None
) -> pl.DataFrame:
    """Preprocess word forms of a language corpus.

    Args:
        file (Path | str): Path to CSV file with pronunciations.
        syll_boundaries (Sequence[str] | str | None, optional): Which symbols should be interpreted as marking syllabic boundaries? Defaults to ".", "ˈ", and "ˌ".

    Returns:
        pl.DataFrame: A Polars dataframe with a word database.
    """
    file = Path(file)

    if syll_boundaries is None:
        syll_boundaries = [".", "ˈ", "ˌ"]
    elif isinstance(syll_boundaries, str):
        syll_boundaries = [syll_boundaries]

    df = (
        pl.scan_csv(file)
        .unique("phon")
        .with_columns(pl.col("phon").str.replace_all(" ", ""))
        .with_columns(
            pl.col("phon")
            .str.replace_many(list(syll_boundaries), ["_"])
            .str.strip_chars("_")
            .alias("phon_tmp")
        )
        .with_columns(
            pl.col("phon_tmp")
            .str.split(r"_")
            .list.eval(pl.element().filter(pl.element() != ""))
            .alias("syll")
        )
        .with_columns(
            pl.col("phon_tmp").str.replace_all("_", "").str.len_chars().alias("n_phon"),
            pl.col("syll").list.len().alias("n_syll"),
        )
        .select(["word", "phon", "syll", "n_phon", "n_syll"])
        .collect()
    )

    return pl.DataFrame(df)


def get_words(langs: pl.DataFrame | None = None, min_words: int = 100):
    if langs is None:
        langs = languages.get_languages()

    words = []

    n_lang = len(langs)
    msg = f"WORDS: {n_lang:,} languages"
    for lang in track(langs["iso639p3code"].to_list(), msg, n_lang):
        try:
            df = utils.add_lang_col(get_lang_words(utils.IN_PATH / f"{lang}.csv"), lang)
            df = df.unique(["word", "phon", "iso639p3code"])

            if len(df) < min_words:
                continue

            words.append(df)
            df.write_parquet(utils.WORD_PATH / f"{lang}.parquet")
        except (FileNotFoundError, pl.exceptions.NoDataError):
            continue

    words = pl.concat(words) if len(words) > 0 else pl.DataFrame()
    words.write_parquet(utils.OUT_PATH / "words.parquet")

    return words


def main():
    langs = pl.read_parquet(utils.OUT_PATH / "languages.parquet")
    return get_words(langs)


if __name__ == "__main__":
    w = main()
