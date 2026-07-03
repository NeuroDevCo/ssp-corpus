from pathlib import Path
from tempfile import tempdir


import polars as pl
from rich.progress import track
from wordfreq import zipf_frequency

from src import languages, utils


@utils.benchmark
def get_frequencies_subtlex(
    x: pl.DataFrame, file: Path = Path("out/frequencies.parquet")
):
    if file.exists():
        return pl.read_parquet(file)

    if len(x) == 0:
        return

    langs_lookup = {
        p3: languages.iso639_p3_to_p1(p3) for p3 in set(x["iso639p3code"].to_list())
    }

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
        f.write(
            ",".join(["word", "iso639p3code", "freq_subtlex", "freq_subtlex_pmw"])
            + "\n"
        )

        for p1, p3, w in track(df.iter_rows(named=False), msg, len(df)):
            if not p1:
                continue
            try:
                freq = zipf_frequency(w, p1)
                freq_pmw = 10 ** (freq - 3)
            except LookupError:
                continue
            if not freq:
                continue
            data = ",".join([w, p3, str(freq), str(freq_pmw) if freq else ""])
            f.write(data + "\n")

    df = pl.read_csv(temp_file).sort(pl.col(["iso639p3code", "freq_subtlex"]))
    df.write_parquet(file)
    return df


def get_frequencies_childes(
    file: Path = Path("out/frequencies-childes.parquet"),
):
    return (
        pl.read_parquet(file)
        .rename({"lang": "iso639p3code", "freq": "freq_childes"})
        .with_columns(freq_childes_pmw=10 ** (pl.col("freq_childes") - 3))
        .sort(pl.col(["iso639p3code", "freq_childes"]))
    )


@utils.benchmark
def get_frequencies(which: str = "subtlex", **kwargs):
    if which not in ["subtlex", "childes"]:
        raise ValueError("which must be 'subtlex' or 'childes'")

    if which == "subtlex":
        return get_frequencies_subtlex(**kwargs)

    return get_frequencies_childes(**kwargs)


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

    freqs_subtlex = get_frequencies(which="subtlex", x=words)

    on = list(set(words.columns).intersection(freqs_subtlex.columns))
    words = words.join(freqs_subtlex, on=on, how="left").join(
        langs.select("iso639p3code"), on="iso639p3code", how="left"
    )

    freqs_childes = get_frequencies(which="childes")
    on = list(set(words.columns).intersection(freqs_childes.columns))
    words = words.join(freqs_childes, on=on, how="left").join(
        langs.select("iso639p3code"), on="iso639p3code", how="left"
    )

    words.write_parquet(utils.OUT_PATH / "words.parquet")

    return words


def main():
    langs = pl.read_parquet(utils.OUT_PATH / "languages.parquet")
    return get_words(langs)


if __name__ == "__main__":
    w = main()
