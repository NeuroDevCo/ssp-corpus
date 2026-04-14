from pathlib import Path
from warnings import warn

from iso639 import Language, LanguageNotFoundError
import numpy as np
from panphon import FeatureTable
from panphon.sonority import Sonority
import polars as pl
from rich.progress import track

IN_PATH: Path = Path("data/raw")
OUT_PATH: Path = Path("data/processed/syllables")


FT = FeatureTable().seg_dict
SON_ALL = Sonority()
SON = {s: SON_ALL.sonority(s) for s in FT if s.strip()}


def count_clusters(x: str, symbol: str = "V") -> int:
    """Count the number of clusters of a particular type of segment in a
    phonological word form.

    Args:
        x (str): Word-form to count the number of clusters of.
        symbol (str, optional): Symbol to look for to identify instances of the segment type. Defaults to "V" (vowels).

    Raises:
        ValueError: If `symbol` is not of length 1.

    Returns:
        Number of clusters of consecutive instances of the segment type in `x`.
    """
    if len(symbol) != 1:
        raise ValueError("`symbol` must of of length 1")

    return list(x).count(symbol)


def has_one_peak(x: list[int] | np.ndarray, allow_flat: bool = False) -> bool:
    """
    Determine whether a sequence contains at most one strict local maximum.

    Notes:
        A strict local maximum is defined as an element that is greater than its immediate neighbours. For the first and last elements, they are considered maxima if they are greater than their only neighbour.

    Returns:
        bool: True if the number of maxima is 0 or 1, False when there are two or more.

    Examples:
        >>> has_only_one_maximum([1, 2, 3, 2, 1])
        True
        >>> has_only_one_maximum([1, 2, 1, 2, 1])
        False
        >>> has_only_one_maximum([5, 5, 5])
        True
        >>> has_only_one_maximum([])
        True
        >>> has_only_one_maximum([42])
        True
    """
    n = len(x)
    if n == 0:
        return True

    count = 0
    for i in range(n):
        left_ok = (i == 0) or (x[i] > x[i - 1])
        right_ok = (i == n - 1) or (x[i] > x[i + 1])
        if left_ok and right_ok:
            count += 1
            if count != 1 and not allow_flat:
                return False
            if count > 1:
                return False
    return True


def classify_segments(x: str, on_missing: str = "skip") -> str | None:
    """
    Classify segments in a word form into consonants (C), vowels (V), and semi-vowels (v).

    Note:
        This function uses Panphon's segment features to classify segments into consonants and vowels. It loops through each segment in the sequence, and makes the following decissions:

        0) If any of the segments is not found in the Panphon feature table, the syllable is skipped.
        1) If the segment is syllabic (can act as syllable nucleus), it is classified as a vowel. If the segment is consonantal, it is classified as a consonant. The iteration is finished.
        2) If the segment is not syllabic nor consonantal, it is considered a glide or semi-vowel. Following Ramus et al. (1999), the following decissions are made in order:
            2.1) If the segment is the first of the sequence, it is classified as a glide, therefore consonant.
            2.2) If the segment is the last of the sequence, it is classified as a semi-vowel.
            2.3) If the previous segment was classified as a vowel, the segment is classified as a semi-vowel.
            2.4) If the previous segment was classified as a consonant, the segment is classified as a glide, therefore a consonant.


    Args:
        x (str): Sequence of segments to classify.
        on_missing (str, optional): What should happen if a segment is not found in the Panphon features table? If "error", an error is raised. If "warning", a warning is displayed and the segment is skipped. If "skip" (default), the segment is simply skipped.

    Return:
        str | None: Sequence of classified segments, None if any segment in the string is not found in the Panphon feature table.
    """
    if on_missing not in ["error", "warning", "skip"]:
        raise ValueError("`on_missing` must be one of 'error', 'warning', or 'skip'")

    out = ""
    for i, segment in enumerate(x):
        features = FT.get(segment)
        if features is None:
            msg = f"Segment {segment} missing from features table"
            if on_missing == "error":
                raise KeyError(msg)
            if on_missing == "warn":
                warn(msg)
            return
        elif features.get("syl") == 1:
            out += "V"
        elif features.get("cons") == 1:
            out += "C"
        else:
            if i == 0:
                out += "C"
            elif i == len(x) - 1:
                out += "v"
            else:
                prev_features = FT.get(x[i - 1])
                if prev_features is None:
                    continue
                elif prev_features.get("syl") == 1:
                    out += "v"
                else:
                    out += "C"

    return out


def extract_sonority(x: str) -> list[int]:
    """Get sonority score (1-9) of the segments of a word-form.

    Args:
        x (str): Sequence of segments to get sonority for.

    Return:
        list[int]: Sequence of sonority values.
    """
    return [SON[s] for s in x if s in SON]


def iso639_p3_to_p1(p3: str):
    try:
        lang = Language.from_part3(p3)
        return str(lang.part1)
    except (LookupError, LanguageNotFoundError, ValueError, AttributeError):
        return


def get_lang_syllables(x: pl.DataFrame) -> pl.DataFrame | None:
    """
    Get syllables from a words database.

    Args:
        x (pl.DataFrame): A Polars dataframe with a word database, as generated by `get_words`.

    Returns:
        pl.DataFrame: A Polars dataframe with a syllables database.
    """

    df = (
        x.explode("syll")
        .with_columns(
            struct=pl.col("syll").map_elements(
                classify_segments, return_dtype=pl.String
            )
        )
        .filter(pl.col("struct").str.count_matches("V") == 1)
        .group_by(["iso639p3code", "syll", "struct"])
        .agg(pl.col("freq_pmw").sum())
        .with_columns(
            freq=(pl.col("freq_pmw") + 0.01).log10() + 3,
            n_phon=pl.col("syll").str.len_chars(),
            sonority=pl.col("syll")
            .str.extract_all(".")
            .list.eval(pl.element().replace(SON)),
        )
        .with_columns(
            is_ssp=pl.col("sonority").map_elements(
                has_one_peak, return_dtype=pl.Boolean
            )
        )
    )

    return df


def get_syllables(words: pl.DataFrame, min_syll: int = 100) -> pl.DataFrame:
    """Main script."""

    syll = []

    part_words = words.partition_by("iso639p3code")
    msg = f"SYLLABLES: {len(part_words):,} languages"
    for lang_words in track(part_words, msg, len(part_words)):
        try:
            df = get_lang_syllables(lang_words)
            if df is None or len(df) < min_syll:
                continue
            syll.append(df)

            lang = lang_words.unique("iso639p3code").get_column("iso639p3code")
            df.write_parquet(OUT_PATH / f"{lang[0]}.parquet")

        except pl.exceptions.NoDataError as e:
            print(e)

    syll = pl.concat(syll) if len(syll) > 0 else pl.DataFrame()
    cols_keep = [
        "iso639p3code",
        "syll",
        "struct",
        "sonority",
        "freq_pmw",
        "freq",
        "is_ssp",
    ]

    syll = syll.sort(["iso639p3code", "syll"]).select(cols_keep)
    syll.write_parquet(Path("out/syllables.parquet"))

    return syll


if __name__ == "__main__":
    langs = pl.read_parquet(Path("out/languages.parquet"))
    words = pl.read_parquet(Path("out/words.parquet"))
    sylls = get_syllables(words)
