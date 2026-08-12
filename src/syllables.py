from pathlib import Path
from warnings import warn

import numpy as np
import polars as pl
from panphon import FeatureTable
from panphon.sonority import Sonority
from rich.progress import track

from src import utils

FT = FeatureTable()
SON_ALL = Sonority()
SON = {s: SON_ALL.sonority(s) for s in FT.seg_dict if s.strip()}


def classify_segments(x: str, on_missing: str = "warning") -> str | None:
    """
    Classify segments in a word form into consonants (C), vowels (V), and semi-vowels (v).

    This function uses Panphon's segment features to classify segments into consonants and vowels. It loops through each segment in the sequence, and makes the following decissions:

    0) If any of the segments is not found in the Panphon feature table, the syllable is skipped.
    1) If the segment is syllabic (can act as syllable nucleus), it is classified as a vowel. If the segment is consonantal, it is classified as a consonant. The iteration is finished.
    2) If the segment is not syllabic nor consonantal, it is considered a glide or semi-vowel. Following Ramus et al. (1999), the following decissions are made in order:
        2.1) If the segment is the first of the sequence, it is classified as a glide, therefore consonant.
        2.2) If the segment is the last of the sequence, it is classified as a semi-vowel.
        2.3) If the previous segment was classified as a vowel, the segment is classified as a semi-vowel.
        2.4) If the previous segment was classified as a consonant, the segment is classified as a glide, therefore a consonant.

    Note:
        Because of the order in which conditions are checked, segments that would normally be classified as consonantal like /n/ may in some contexts be classified as vocalic based on their syllabicity (can act as syllabic nucleus). This is the case for /n/ in English "cotton" /ˈkɒt.n̩/.

    Args:
        x (str): Sequence of segments to classify.
        on_missing (str, optional): What should happen if a segment is not found in the Panphon features table? If "error", an error is raised. If "warning", a warning is displayed and the segment is skipped. If "skip" (default), the segment is simply skipped.

    Return:
        str | None: Sequence of classified segments, None if any segment in the string is not found in the Panphon feature table.
    """
    if on_missing not in ["error", "warning", "skip"]:
        raise ValueError("`on_missing` must be one of 'error', 'warning', or 'skip'")

    out = ""
    for i, segment in enumerate(FT.ipa_segs(x)):
        features = FT.seg_dict.get(segment)

        if features is None:
            msg = f"Segment {segment} missing from features table"
            if on_missing == "error":
                raise KeyError(msg)
            if on_missing == "warning":
                warn(msg)
            return

        elif features.get("syl") == 1:
            out += "V"  # vowel (syllabic nucleus)

        elif features.get("cons") == 1:
            out += "C"  # consonant (onset or coda)

        else:
            if i == 0:
                out += "C"  # glide (onset or coda)
            elif i == len(x) - 1:
                out += "v"  # semi-vowel (syllabic nucleus as part of diphtong)
            else:
                # unclear glide/semi-vowel classification, see previous segment
                prev_features = FT.seg_dict.get(x[i - 1])
                if prev_features is None:
                    continue
                elif prev_features.get("syl") == 1:
                    out += "v"  # after V, semi-vowel
                else:
                    out += "C"  # glide otherwise

    return out


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


def extract_sonority(x: str) -> list[int]:
    """Get sonority score (1-9) of the segments of a word-form.

    Args:
        x (str): Sequence of segments to get sonority for.

    Return:
        list[int]: Sequence of sonority values.
    """
    return [SON[s] for s in FT.ipa_segs(x) if s in SON]


def has_one_peak(x: list[int] | np.ndarray, allow_flat: bool = True) -> bool:
    """
    Determine whether a sequence contains at most one strict local maximum.

    Notes:
        A strict local maximum is defined as an element that is greater than its immediate neighbours. For the first and last elements, they are considered maxima if they are greater than their only neighbour.

    Args:
        x (list[int] | np.ndarray): Sequence of integers to evaluate.
        allow_flat (bool, optional): Should consecutive identical elements be considered truthy or falsy? Defaults to True.

    Returns:
        bool: True if the number of maxima is 0 or 1, False when there are two or more.

    Examples:
        >>> has_one_peak([1, 2, 3, 2, 1])
        True
        >>> has_one_peak([1, 2, 1, 2, 1])
        False
        >>> has_one_peak([5, 5, 5])
        True
        >>> has_one_peak([])
        True
        >>> has_one_peak([42])
        True
        >>> has_one_peak([2, 2, 1], allow_flat = True)
        True
        >>> has_one_peak([2, 2, 1], allow_flat = False)
        False
    """
    n = len(x)

    if n == 0:
        return True

    count = 0
    for i in range(n):
        if allow_flat:
            left_ok = (i == 0) or (x[i] > x[i - 1])
            right_ok = (i == n - 1) or (x[i] > x[i + 1])
        else:
            left_ok = (i == 0) or (x[i] >= x[i - 1])
            right_ok = (i == n - 1) or (x[i] >= x[i + 1])

        if left_ok and right_ok:
            count += 1

            if count != 1 and not allow_flat:
                return False

            if count > 1:
                return False
    return True


def get_lang_syllables(
    x: pl.DataFrame, freq: bool = True, allow_flat: bool = True
) -> pl.DataFrame | None:
    """
    Get syllables from a words database.

    Args:
        x (pl.DataFrame): A Polars dataframe with a word database, as generated by `get_words`.
        allow_flat (bool, optional): Should consecutive identical elements in sonority sequence be considered truthy or falsy? Defaults to True.

    Returns:
        pl.DataFrame: A Polars dataframe with a syllables database.
    """
    if len(x) == 0:
        return

    if freq:
        x = x.filter(pl.col("freq_pmw").is_not_null())

    df = (
        x.explode("syll")
        .unique(["iso639p3code", "syll"])
        .with_columns(
            struct=pl.col("syll").map_elements(
                classify_segments, return_dtype=pl.String
            )
        )
        .filter(pl.col("struct").str.count_matches("V") == 1)
    )

    if freq:
        df = (
            df.group_by(["iso639p3code", "syll", "struct"])
            .agg(pl.col("freq_pmw").sum())
            .with_columns(freq=(pl.col("freq_pmw") + 0.01).log10() + 3)
        )
    else:
        df = df.unique(["syll"])

    df = df.with_columns(
        n_phon=pl.col("syll").str.len_chars(),
        sonority=pl.col("syll")
        .map_elements(FT.ipa_segs, return_dtype=pl.List(pl.String))
        .list.eval(pl.element().replace(SON)),
    ).with_columns(
        is_ssp=pl.col("sonority").map_elements(
            lambda x: has_one_peak(x, allow_flat=allow_flat), return_dtype=pl.Boolean
        ),
    )

    return df


def get_syllables(
    words: pl.DataFrame,
    min_syll: int = 100,
    freq: bool = True,
    allow_flat: bool = True,
    path: Path | None = None,
) -> pl.DataFrame:
    """Main script."""

    syll = []

    part_words = words.partition_by("iso639p3code")
    msg = f"SYLLABLES: {len(part_words):,} languages"
    for lang_words in track(part_words, msg, len(part_words)):
        try:
            df = get_lang_syllables(lang_words, freq=freq, allow_flat=allow_flat)
            if df is None or len(df) < min_syll:
                continue
            syll.append(df)

            lang = lang_words.unique("iso639p3code").get_column("iso639p3code")
            df.write_parquet(utils.SYLL_PATH / f"{lang[0]}.parquet")

        except pl.exceptions.NoDataError as e:
            print(e)

    syll = pl.concat(syll) if len(syll) > 0 else pl.DataFrame()
    cols_keep = ["iso639p3code", "syll", "struct", "sonority", "is_ssp"]

    if freq:
        cols_keep += ["freq"]

    syll = syll.sort(["iso639p3code", "syll"]).select(cols_keep)

    if path:
        syll.write_parquet(path)

    return syll


def main():
    w = pl.read_parquet(utils.RESULTS_PATH / "words.parquet")

    s_file = utils.RESULTS_PATH / "syllables.parquet"
    s = get_syllables(w, path=s_file)

    sf_file = utils.RESULTS_PATH / "no-flat" / "syllables.parquet"
    sf = get_syllables(w, allow_flat=False, path=sf_file)
    return s, sf


if __name__ == "__main__":
    s, sf = main()
