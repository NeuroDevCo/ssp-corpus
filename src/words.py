import logging
from pathlib import Path

import iso639
import polars as pl
import wordfreq
from rich.progress import track

from src import languages, utils

SCHEMA = {"word": pl.String, "phon": pl.String}
FREQ_SCHEMA = {
    "word": pl.String,
    "freq": pl.Float64,
    "freq_pmw": pl.Float64,
}

logger = logging.getLogger(__name__)


def remove_tones(
    x: str,
    tone_chars: str = "˩˨˧˦˥",
    tone_diacritics: str = "\u0301\u0300\u0302\u030c",
) -> str:
    """Remove tone annotations from phonological transcription.

    Args:
        x (str): Phonological transcription.
        tone_chars (str, optional): Characters with tone annotations to remove. Defaults to "˩˨˧˦˥".
        tone_diacritics (str, optional): Characters with tone diacritic annotations to remove. Defaults to "\u0301\u0300\u0302\u030c".

    Returns:
        str: Phonological transcription with remove tone annotations.
    """
    y = x
    for char in tone_chars + tone_diacritics:
        y = y.replace(char, "")

    return y


@utils.benchmark
@utils.log_shape
def get_frequencies(x: pl.DataFrame) -> pl.DataFrame:
    """Get word frequencies for a dataset using wordfreq.

    Args:
        x (pl.DataFrame): Dataset with language and words to retrieve.

    Raises:
        ValueError: If more than one language is detected in "iso639p3code" column in `x`.

    Returns:
        pl.DataFrame: Dataset of frequencies for available words in wordfreq.
    """
    if len(x) == 0:
        return pl.DataFrame(FREQ_SCHEMA)

    lang = x["iso639p3code"].unique()

    if len(lang) > 1:
        raise ValueError(f"More than one language detected: {lang}")
    lang = lang[0]

    freqs, freq_words = [], []
    for word in x["word"].unique():
        try:
            freqs.append(wordfreq.zipf_frequency(word, lang))
            freq_words.append(word)
        except LookupError:
            continue
        except iso639.LanguageNotFoundError as e:
            logger.warning(f"{lang}: {e}")
            return pl.DataFrame(schema=FREQ_SCHEMA)

    if not freqs:
        logger.warning(f"{lang}: empty frequencies dataset")
        return pl.DataFrame(schema=FREQ_SCHEMA)

    df = (
        pl.DataFrame({"word": freq_words, "freq": freqs})
        .with_columns(freq_pmw=10 ** (pl.col("freq") - 3))
        .sort("freq")
    )
    return df


@utils.with_seps(["\t", ","])
def get_lang_words(
    file: Path | str,
    syll_boundaries: list[str] | None = None,
    tone_char: str = "˩˨˧˦˥",
    sep=",",
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

    df = (
        pl.read_csv(file, separator=sep, has_header=False, schema=SCHEMA)
        .unique("phon")
        .with_columns(
            pl.col("phon")
            .str.replace_many(list(tone_char), [""] * len(tone_char))
            .str.replace_all(" ", "")
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
        # .collect()
    )

    return pl.DataFrame(df)


@utils.log_shape
def get_words(langs: pl.DataFrame | None = None, min_words: int = 100) -> pl.DataFrame:
    """Get words dataset from Wiktionary output files.

    Args:
        langs (pl.DataFrame | None, optional): Dataset of languages to get words for. Defaults to None.
        min_words (int, optional): Minimum available words for a language to be included. Defaults to 100.

    Returns:
        pl.DataFrame: Words dataset.
    """
    if langs is None:
        langs = languages.get_languages()

    words = []
    csv = [p for p in Path("data/raw").glob("*.csv")]
    tsv = [p for p in Path("data/raw").glob("*.tsv")]
    files = csv + tsv
    n_lang = len(langs["iso639p3code"])
    msg = f"WORDS: {n_lang:,} files"

    for lang in track(langs["iso639p3code"], msg, n_lang):
        lang_df = []

        for file in [f for f in files if f.name.startswith(lang)]:
            mode = file.with_suffix("").name.split("-")[-1] if "-" in file.name else ""
            try:
                df = get_lang_words(file)
                df.insert_column(0, pl.Series("iso639p3code", [lang] * len(df)))
                df.insert_column(1, pl.Series("mode", [mode] * len(df)))

                if len(df) < min_words:
                    continue

                freq = get_frequencies(df)
                df = df.join(freq, on=["word"], how="inner")

                lang_df.append(df)
            except (FileNotFoundError, RuntimeError) as e:
                logger.error(f"{lang}: {e}")
                continue
            except pl.exceptions.NoDataError:
                continue

        if not lang_df:
            continue

        lang_df = pl.concat(lang_df).unique(["word", "phon", "iso639p3code"])

        utils.WORD_PATH.mkdir(exist_ok=True, parents=True)
        lang_df.write_parquet(utils.WORD_PATH / f"{lang}.parquet")

        words.append(lang_df)

    words = pl.concat(words) if len(words) > 0 else pl.DataFrame()

    words.write_parquet(utils.RESULTS_PATH / "words.parquet")

    return words


def main():
    """Main script."""
    l = pl.read_parquet(utils.RESULTS_PATH / "languages.parquet")

    return get_words(l)


if __name__ == "__main__":
    w = main()
