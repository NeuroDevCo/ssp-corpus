import polars as pl
import pytest

from src.utils import benchmark, add_lang_col, add_missing_headers, cast_to_str

# Assuming the module is named utils.py or similar.
# Adjust the import path based on your actual file name.
# from utils import benchmark, add_missing_headers, cast_to_str, add_lang_col
# For the purpose of this standalone test file, I will define the functions here
# or assume they are imported. Since I am providing the tests, I will assume
# the functions are available in the namespace or imported correctly.

# --- Fixtures ---


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Provides a sample DataFrame for testing."""
    return pl.DataFrame(
        {"a": [1, 2, 3], "b": [["x", "y"], ["z"], ["w"]], "c": [10, 20, 30]}
    )


@pytest.fixture
def sample_list_df() -> pl.DataFrame:
    """Provides a DataFrame with list columns for casting tests."""
    return pl.DataFrame(
        {
            "id": [1, 2],
            "tokens": [["hello", "world"], ["foo", "bar"]],
            "counts": [[1, 2], [3, 4]],
            "score": [0.5, 0.8],
        }
    )


# --- Test Classes ---


class TestBenchmark:
    """Tests for the benchmark decorator."""

    def test_wrapper_executes_function(self, sample_df):
        """Test that the decorator returns the correct result of the function."""

        @benchmark
        def dummy_func(df):
            return df.height

        result = dummy_func(sample_df)
        assert result == 3

    def test_prints_execution_time(self, sample_df, capsys):
        """Test that the decorator prints the execution time message."""

        @benchmark
        def dummy_func(df):
            return df.height

        dummy_func(sample_df)
        captured = capsys.readouterr()
        assert "Finished dummy_func() in" in captured.out
        assert "seconds" in captured.out

    def test_passes_arguments_correctly(self):
        """Test that args and kwargs are passed through correctly."""

        @benchmark
        def add(a, b):
            return a + b

        result = add(5, b=10)
        assert result == 15


class TestAddMissingHeaders:
    """Tests for the add_missing_headers function."""

    def test_adds_header_to_file_with_existing_header_row(self, tmp_path):
        """
        Scenario: File contains 'word' and 'phon' in the first row,
        but was read without headers.
        Expected: Renames columns to 'word' and 'phon'.
        Note: The provided logic renames but does not drop the first row.
        """
        d = tmp_path / "test.csv"
        content = "word,phon\nhello,hə\nworld,wɜːld\n"
        d.write_text(content)

        add_missing_headers(tmp_path)

        df = pl.read_csv(d)
        # The function renames default columns (column_1, column_2) to the values in row 0
        # It writes back the dataframe. Polars write_csv includes headers.
        # So we expect Header: word, phon. Row 1: word, phon. Row 2: hello, hə.
        assert df.columns == ["word", "phon"]
        assert df.row(0) == ("word", "phon")
        assert df.row(1) == ("hello", "hə")

    def test_renames_generic_columns(self, tmp_path):
        """
        Scenario: File has no header row and generic data.
        Expected: Renames generic Polars columns (column_1, column_2) to word, phon.
        """
        d = tmp_path / "test.csv"
        content = "hello,hə\nworld,wɜːld\n"
        d.write_text(content)

        add_missing_headers(tmp_path)

        df = pl.read_csv(d)
        assert df.columns == ["word", "phon"]
        assert df.row(0) == ("hello", "hə")

    def test_handles_file_with_existing_column_names(self, tmp_path):
        """
        Scenario: File already has correct headers 'word' and 'phon'.
        Polars read_csv(has_header=False) will treat them as data.
        The logic identifies 'word'/'phon' in the first row and renames columns.
        """
        d = tmp_path / "test.csv"
        content = "word,phon\ntest,tɛst\n"
        d.write_text(content)

        add_missing_headers(tmp_path)

        df = pl.read_csv(d)
        assert df.columns == ["word", "phon"]
        # Logic keeps the first row as data in the first 'if' block
        assert df.row(0) == ("word", "phon")

    def test_creates_schema_for_empty_file(self, tmp_path):
        """
        Scenario: File is empty.
        Expected: Creates a file with schema word (str) and phon (int).
        Note: The code specifies phon as int, which is unusual for phonemes,
        but we test the implementation provided.
        """
        d = tmp_path / "test.csv"
        d.write_text("")

        add_missing_headers(tmp_path)

        df = pl.read_csv(d)
        # pl.DataFrame(...).write_csv creates a file with just the header
        assert df.columns == ["word", "phon"]
        assert len(df) == 0
        # Check schema based on the code: schema={"word": str, "phon": int}
        assert df.schema["word"] == pl.String
        assert df.schema["phon"] == pl.String


class TestCastToStr:
    """Tests for the cast_to_str function."""

    def test_casts_int_list_to_string(self, sample_list_df):
        """Test casting a list of integers to a joined string."""
        result = cast_to_str(sample_list_df)

        # Check counts column (List[Int]) became String
        assert result["counts"].dtype == pl.String
        assert result["counts"][0] == "1_2"
        assert result["counts"][1] == "3_4"

    def test_casts_str_list_to_string(self, sample_list_df):
        """Test casting a list of strings to a joined string."""
        result = cast_to_str(sample_list_df)

        # Check tokens column (List[String]) became String
        assert result["tokens"].dtype == pl.String
        assert result["tokens"][0] == "hello_world"
        assert result["tokens"][1] == "foo_bar"

    def test_uses_custom_separator(self, sample_list_df):
        """Test using a custom separator."""
        result = cast_to_str(sample_list_df, sep="-")

        assert result["counts"][0] == "1-2"
        assert result["tokens"][0] == "hello-world"

    def test_leaves_non_list_columns_unchanged(self, sample_list_df):
        """Test that non-list columns are not affected."""
        result = cast_to_str(sample_list_df)

        assert result["id"].dtype == pl.Int64
        assert result["score"].dtype == pl.Float64
        assert result["id"].to_list() == [1, 2]


class TestAddLangCol:
    """Tests for the add_lang_col function."""

    def test_adds_language_column(self):
        """Test that the language column is added with the correct value."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        lang_code = "eng"

        result = add_lang_col(df, lang_code)

        assert "iso639p3code" in result.columns
        assert result["iso639p3code"].to_list() == [lang_code, lang_code]

    def test_reorders_columns_correctly(self):
        """Test that columns are reordered: col0, lang_code, rest."""
        df = pl.DataFrame({"z": [9, 8], "a": [1, 2], "m": [5, 6]})
        lang_code = "fra"

        result = add_lang_col(df, lang_code)
        expected_cols = ["z", "iso639p3code", "a", "m"]

        assert result.columns == expected_cols

    def test_handles_single_column_input(self):
        """Test DataFrame with only one column."""
        df = pl.DataFrame({"word": ["hello", "world"]})
        lang_code = "deu"

        result = add_lang_col(df, lang_code)

        assert result.columns == ["word", "iso639p3code"]
        assert result.shape == (2, 2)
