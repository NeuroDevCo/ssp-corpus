from collections.abc import Sequence
from pathlib import Path

from iso639 import Language
import polars as pl
from rich.progress import track
from wordfreq import zipf_frequency
from src import languages
from src.utils import add_lang_col, benchmark, cast_to_str

IN_PATH: Path = Path("data/raw")
OUT_PATH: Path = Path("data/processed/words")


def iso639_p3_to_p1(p3: str):
    try:
        lang = Language.from_part3(p3)
        return str(lang.part1)
    except (LookupError, ValueError, AttributeError):
        return


@benchmark
def get_frequencies(
    x: pl.DataFrame,
    file: Path | str = Path("out/frequencies.parquet"),
    reload: bool = True,
    **kwargs,
):

    file = Path(file)
    if file.exists() and reload:
        return pl.read_parquet(file)

    words = x["word"].to_list()
    lang_p3 = x["iso639p3code"].to_list()
    lang_p1 = [iso639_p3_to_p1(p3) for p3 in lang_p3]

    def get_freq(word: str, lang: str, **kwargs):
        try:
            return zipf_frequency(word, lang, **kwargs)
        except LookupError:
            return

    freqs = [get_freq(w, lang) for w, lang in zip(words, lang_p1) if lang]

    df = pl.DataFrame({"lang": lang_p3, "word": words, "freq": freqs})
    df.write_parquet(file)
    return df


def get_lang_words(
    file: Path | str, syll_boundaries: Sequence[str] | str | None = None
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
        syll_boundaries = [r".", "ˈ", "ˌ"]
    elif isinstance(syll_boundaries, str):
        syll_boundaries = [syll_boundaries]

    df = (
        pl.scan_csv(file)
        .unique("phon")
        .with_columns(
            pl.col("phon")
            .str.replace_all(" ", "")
            .str.replace_many({b: "." for b in syll_boundaries})
            .str.strip_chars("."),
            pl.col("phon")
            .str.split(".")
            .list.eval(pl.element().filter(pl.element() != ""))
            .alias("syll"),
            pl.col("phon").str.len_chars().alias("n_phon"),
        )
        .with_columns(
            pl.col("syll").list.len().alias("n_syll"),
        )
        .collect()
    )

    return pl.DataFrame(df)


def get_words(x: pl.DataFrame, min_words: int = 100):
    words = []

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    msg = f"WORDS: Preprocessing {len(x)} languages"
    for lang in track(x["iso639p3code"].to_list(), msg, len(x)):
        try:
            w = add_lang_col(get_lang_words(IN_PATH / f"{lang}.csv"), lang)
            if len(w) < min_words:
                continue
            words.append(w)
            cast_to_str(w).write_parquet(OUT_PATH / f"{lang}.parquet")
        except (FileNotFoundError, pl.exceptions.NoDataError):
            continue

    return pl.concat(words) if len(words) > 0 else pl.DataFrame()


if __name__ == "__main__":
    LANGS = languages.get_languages()
    WORDS = get_words(LANGS)
    FREQS = get_frequencies(WORDS)
