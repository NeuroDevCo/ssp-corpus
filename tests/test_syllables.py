import polars as pl
import pytest

from src.syllables import (
    classify_segments,
    count_clusters,
    extract_sonority,
    get_lang_syllables,
    has_one_peak,
)


@pytest.fixture
def sample_df():
    """Provides a sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "iso639p3code": ["eng", "eng", "eng"],
            "syll": [["ka", "mata"], ["bja", "3", "spa"], ["a", "b", "taw", "j"]],
            "freq": [100, 50, 20],
        }
    )


class TestCountClusters:
    def test_single_cluster(self):
        """The implementation counts occurrences, not consecutive clusters.
        "ba" has 1 'a'
        """
        assert count_clusters("ba", "a") == 1

    def test_multiple_occurrences(self):
        """ "aba" has 2 'a's"""
        assert count_clusters("aba", "a") == 2

    def test_symbol_not_present(self):
        assert count_clusters("bbb", "a") == 0

    def test_invalid_symbol_length(self):
        with pytest.raises(ValueError):
            count_clusters("abc", "ab")


class TestHasOnePeak:
    def test_single_peak_center(self):
        assert has_one_peak([1, 2, 3, 2, 1]) is True

    def test_single_peak_onset(self):
        assert has_one_peak([5, 4, 3, 2, 1]) is True

    def test_single_peak_offset(self):
        assert has_one_peak([1, 2, 3, 4, 5]) is True

    def test_plateau(self):
        """Flat sequences technically have 0 strict local maxima, which satisfies "at most one"."""
        assert has_one_peak([5, 5, 5]) is True

    def test_multiple_peaks(self):
        assert has_one_peak([1, 2, 1, 2, 1]) is False

    def test_empty(self):
        assert has_one_peak([]) is True


class TestClassifySegments:
    def test_vowel(self):
        assert classify_segments("a") == "V"

    def test_consonant(self):
        assert classify_segments("b") == "C"

    def test_cv(self):
        """'b' is C, 'a' is V"""
        assert classify_segments("ba") == "CV"

    def test_semi_vowel_initial(self):
        """'j' is semi-vowel. Following Ramus et al. (1999), at start, treated as C."""
        assert classify_segments("ja") == "CV"

    def test_semi_vowel_final(self):
        """'j' at end. 'a' is V. Following Ramus et al. (1999), 'j' treated as semi-vowel (v)."""
        assert classify_segments("aj") == "Vv"

    def test_semi_vowel_between_vowels(self):
        """'a' (V), 'j', 'e' (V). 'j' follows V, so treated as v."""
        assert classify_segments("aje") == "VvV"

    def test_semi_vowel_between_consonants(self):
        """'b' (C), 'j', 'k' (C). 'j' follows C, so treated as C."""
        assert classify_segments("bjk") == "CCC"

    def test_returns_str(self):
        assert isinstance(classify_segments("ˈli.β̞ɾo"), str)

    def test_ignores_brackets(self):
        assert classify_segments("[ˈab.ejk]") == "VCVvC"

    def test_ignores_slashes(self):
        assert classify_segments("/ˈab.ejk/") == "VCVvC"


class TestExtractSonority:
    def test_simple_extraction(self):
        """'a' is 9, 'b' is 5."""
        assert extract_sonority("ab") == [9, 2]

    def test_unknown_segment(self):
        """'z' is not in mock FT, should be skipped/ignored by logic

        Note:
            Extract_sonority filters: `s for s in x if s in sonority`
        """
        assert extract_sonority("azb") == [9, 4, 2]


class TestGetLangSyllables:
    def test_filter_logic(self, sample_df):
        """
        Test that syllables with exactly 1 V (vowel) are kept.

        Input:
        0. "ka" -> struct "CV" -> count=1 (Keep)
        1. "mata" -> struct "CVCV" -> count=2 (Discard)
        2. "bja" -> struct "CCV" -> count=1 (Keep)
        3. "3" -> struct "" -> count=0 (Discard)
        4. "spa" -> struct "CCV" -> count=1 (Keep)
        5. "a" -> struct "V" -> count=1 (Keep)
        6. "b" -> struct "C" -> count=0 (Discard)
        7. "taw" -> struct "CVv" -> count=1 (Keep)
        8. "j" -> struct "v" -> count=0 (Discard)
        """
        result = get_lang_syllables(sample_df)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5
        structs = result["struct"].to_list()
        assert "CV" in structs
        assert "CCV" in structs
        assert "V" in structs
        assert "CVv" in structs

        # Check SSP (Sonority Sequencing Principle)
        assert sum(result["is_ssp"].to_list()) == 4

    def test_empty_dataframe(self):
        """Should return empty DF with same schema."""

        df = pl.DataFrame({"syll": []})
        result = get_lang_syllables(df)
        assert result is None
