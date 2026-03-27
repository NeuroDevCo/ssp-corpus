from pathlib import Path
from tempfile import tempdir

from iso639 import Language
import polars as pl
from rich.progress import track
from wordfreq import zipf_frequency

from src.languages import get_languages
from src.utils import add_lang_col, benchmark

IN_PATH: Path = Path("data/raw")
OUT_PATH: Path = Path("data/processed/words")


def iso639_p3_to_p1(p3: str):
    try:
        lang = Language.from_part3(p3)
        return str(lang.part1)
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


@benchmark
def get_frequencies(
    x: pl.DataFrame,
    file: Path | str = Path("out/frequencies.parquet"),
    reload: bool = True,
    buffer_size: int = 10_000,
    **kwargs,
):
    file = Path(file)
    if file.exists():
        return pl.read_parquet(file)

    langs_lookup = {p3: iso639_p3_to_p1(p3) for p3 in set(x["iso639p3code"].to_list())}

    df = (
        x.with_columns(
            pl.col("iso639p3code").replace(langs_lookup).alias("iso639p1code")
        )
        .select(["iso639p1code", "iso639p3code", "word"])
        .unique(["iso639p3code", "word"])
    )

    temp_file = Path(str(tempdir)) / "frequencies.txt"
    msg = f"FREQS: {len(df):,} words"
    with temp_file.open("w+") as f:
        f.write(",".join(["word", "iso639p3code", "freq"]) + "\n")

        for p1, p3, w in track(df.iter_rows(named=False), msg, len(df)):
            if not p1:
                continue
            try:
                freq = zipf_frequency(w, p1)
            except LookupError:
                continue
            if not freq:
                continue
            data = ",".join([w, p3, str(freq) if freq else ""])
            f.write(data + "\n")

    df = pl.read_csv(temp_file).sort(pl.col(["iso639p3code", "freq"]))
    df.write_parquet(file)
    return df


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


def get_words(
    languages: pl.DataFrame | None = None,
    add_frequencies: bool = True,
    min_words: int = 100,
):
    if languages is None:
        languages = get_languages()

    words = []

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    n_lang = len(languages)
    msg = f"WORDS: {n_lang:,} languages"
    for lang in track(languages["iso639p3code"].to_list(), msg, n_lang):
        try:
            df = add_lang_col(get_lang_words(IN_PATH / f"{lang}.csv"), lang)
            df = df.unique(["word", "phon", "iso639p3code"])

            if len(df) < min_words:
                continue
            words.append(df)
            df.write_parquet(OUT_PATH / f"{lang}.parquet")
        except (FileNotFoundError, pl.exceptions.NoDataError):
            continue

    words = pl.concat(words) if len(words) > 0 else pl.DataFrame()

    if add_frequencies:
        freqs = get_frequencies(words)
        on = list(set(words.columns).intersection(freqs.columns))
        words = words.join(freqs, on=on, how="left")

    words.write_parquet(Path("out/words.parquet"))

    return words


if __name__ == "__main__":
    langs = get_languages()
    words = get_words(langs)
    freqs = get_frequencies(words)
