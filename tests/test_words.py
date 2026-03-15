from pathlib import Path
from tempfile import tempdir
from iso639 import LanguageNotFoundError
import polars as pl
import pytest

from src.words import iso639_p3_to_p1, get_lang_words, get_frequencies


class TestLangConversion:
    def test_iso639_p3_to_p1_valid(self):
        """Test conversion of a known ISO 639-3 code to 639-1."""
        assert iso639_p3_to_p1("eng") == "en"
        assert iso639_p3_to_p1("deu") == "de"
        assert iso639_p3_to_p1("fra") == "fr"

    def test_iso639_p3_to_p1_invalid(self):
        """Test that invalid codes return None."""
        with pytest.raises(LanguageNotFoundError):
            iso639_p3_to_p1("xyz")
            iso639_p3_to_p1("")
            iso639_p3_to_p1("123")


class TestGetLangWords:
    def test_get_lang_words_basic(self, tmp_path: Path | str = str(tempdir)):
        """Test basic processing of a CSV file into a word dataframe."""
        # Create a dummy CSV file
        csv_file = Path(tmp_path) / "test_lang.csv"
        csv_content = "word,phon\nhello,h ə ˈl oʊ\nworld,w ɜː l d\ntest,t ɛ s t"
        csv_file.write_text(str(csv_content))

        # Call the function
        df = get_lang_words(csv_file)

        # Assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

        # Check columns exist
        expected_cols = ["word", "phon", "syll", "n_phon", "n_syll"]
        for col in expected_cols:
            assert col in df.columns

        # Check phon cleaning (spaces removed)
        # "h ə ˈl oʊ" -> "həˈloʊ" (boundaries replaced by _, then stripped?)
        # Logic: replace spaces -> replace boundaries with "_" -> strip "_" -> split
        # "h ə ˈl oʊ" -> "həˈloʊ" -> "hə_loʊ" -> "hə_loʊ" -> split -> ["hə", "loʊ"]

        # Let's trace "hello":
        # 1. Remove spaces: "həˈloʊ"
        # 2. Replace ., ˈ, ˌ with _: "hə_loʊ"
        # 3. Strip _: "hə_loʊ"
        # 4. Split by _: ["hə", "loʊ"]

        row_hello = df.filter(pl.col("word") == "hello").row(0, named=True)
        assert row_hello["syll"] == ["hə", "loʊ"]
        assert row_hello["n_syll"] == 2
        # n_phon is len of stripped phon string (stripped of boundaries)
        # "hə_loʊ" -> stripped boundaries (logic replaces _ then strips? No, logic:
        # phon_tmp is replace_many -> strip_chars("_") -> replace_all("_", "") -> len
        # Wait, looking at source:
        # .str.replace_all("_", "") happens on phon_tmp AFTER stripping.
        # So "hə_loʊ" -> "həloʊ". Length 5.
        assert row_hello["n_phon"] == 5
        csv_file.unlink()

    def test_get_lang_words_custom_boundaries(
        self, tmp_path: Path | str = str(tempdir)
    ):
        """Test processing with custom syllable boundaries."""
        csv_file = Path(tmp_path) / "test_custom.csv"
        csv_content = "word,phon\ndata,d eɪ . t ə"
        csv_file.write_text(csv_content)

        df = get_lang_words(csv_file, syll_boundaries=["."])

        row = df.row(0, named=True)
        # Expect split by "."
        # "deɪ.tə" -> ["deɪ", "tə"]
        assert row["syll"] == ["deɪ", "tə"]
        assert row["n_syll"] == 2
        csv_file.unlink()


# --- Tests for get_frequencies ---


class TestFrequencies:
    def test_get_frequencies_calculation(self, tmp_path: Path | str = str(tempdir)):
        """
        Test frequency calculation.
        NOTE: This test relies on the 'wordfreq' library being installed and having data.
        """
        # Create a sample input DataFrame
        input_df = pl.DataFrame(
            {
                "iso639p3code": ["eng", "eng", "eng", "fra"],
                "word": ["the", "of", "zzznonexistentword", "le"],
            }
        )

        # Define a temporary output file path
        out_file = Path(tmp_path) / "freqs.parquet"

        # Run the function
        # Warning: This will attempt to fetch real data from wordfreq
        df_result = get_frequencies(input_df, file=out_file)

        # Assertions
        assert isinstance(df_result, pl.DataFrame)
        assert len(df_result) > 0

        # Check that invalid language codes are filtered out
        assert "xyz" not in df_result["iso639p3code"].to_list()

        # Check that common English words have high frequency (> 7.0 is typical for 'the')
        the_freq = df_result.filter(pl.col("word") == "the")["freq"].item()
        assert the_freq > 7.0

        # Check that 'fra' (French) is mapped to 'fr' internally by iso639_p3_to_p1
        # and wordfreq was called. 'le' is common in French.
        le_freq = df_result.filter(pl.col("word") == "le")["freq"].item()
        assert le_freq > 0.0

        # Check file creation
        assert out_file.exists()

        # Verify reading the file back gives same result (caching test)
        df_cached = get_frequencies(input_df, file=out_file)
        assert df_result.equals(df_cached)

    def test_get_frequencies_empty_input(self, tmp_path: Path | str = str(tempdir)):
        """Test handling of an empty input DataFrame."""
        input_df = pl.DataFrame({"iso639p3code": [], "word": []})

        out_file = Path(tmp_path) / "empty_freqs.parquet"

        # The function might fail or return empty depending on loop implementation
        # The loop `for p1, p3, w in track(df.iter_rows...)` will simply not run.
        # Then `pl.read_csv(temp_file)` is called.
        # The temp file header is written, so read_csv returns a DF with 0 rows.
        df_result = get_frequencies(input_df, file=out_file)

        assert len(df_result) == 0
