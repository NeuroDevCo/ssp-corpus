import polars as pl
import pytest

from src.languages import get_glottolog, get_languages


@pytest.fixture
def mock_glottolog_trees_data():
    """Mock data for Glottolog trees.csv."""
    return pl.DataFrame(
        {
            "ID": ["fam1", "fam2"],
            "Description": [
                "Glottolog classification of the languages in the family Indo-European",
                "Glottolog classification of the languages in the family Austronesian",
            ],
        }
    ).lazy()


@pytest.fixture
def mock_glottolog_langs_data():
    """Mock data for Glottolog languages.csv."""
    return pl.DataFrame(
        {
            "ID": ["lang1", "lang2", "lang3", "lang4"],
            "Name": ["English", "Latin", "IsolateX", "UnknownFam"],
            "Family_ID": ["fam1", "fam1", "fam2", None],
            "ISO639P3code": ["eng", "lat", "iso", None],  # None should be dropped
            "Level": [
                "language",
                "language",
                "language",
                "dialect",
            ],  # dialect should be dropped
            "Glottocode": ["stan1293", "lati1261", "isol1234", "unkn1234"],
            "Latitude": [10.0, 20.0, 30.0, 40.0],
            "Longitude": [11.0, 21.0, 31.0, 41.0],
            "is_isolate": [False, False, True, False],
        }
    ).lazy()


class TestGetGlottolog:
    """Tests for the get_glottolog function."""

    def test_filters_and_transforms_data(self):
        """
        Test that languages are filtered by Level, nulls are dropped,
        columns are lowercased, and isolates are handled correctly.
        """
        df = get_glottolog()

        assert isinstance(df, pl.DataFrame)

        assert "iso639p3code" in df.columns
        assert "glottocode" in df.columns
        assert df.height

        # Basque is an isolate. family_id should be "Isolate"
        isolate_row = df.filter(pl.col("iso639p3code") == "eus").row(0, named=True)
        assert isolate_row["family_id"] == "Isolate"

        # Catalan is a non-isolate logic
        eng_row = df.filter(pl.col("iso639p3code") == "cat").row(0, named=True)
        assert eng_row["family_id"] == "Indo-European"

        expected_cols = [
            "glottocode",
            "lang",
            "family_id",
            "family_name",
            "iso639p3code",
            "latitude",
            "longitude",
        ]
        assert df.columns == expected_cols


class TestGetLanguages:
    """Tests for the get_languages function."""

    def test_joins_and_saves_data(self, tmp_path):
        """Test that glottolog and phoible are joined and saved to parquet."""

        # Define output file
        out_file = tmp_path / "languages.parquet"

        # Execute
        result = get_languages(file=out_file)

        # Assertions
        assert isinstance(result, pl.DataFrame)

        # Check sorting
        assert result.to_dicts() == sorted(
            result.to_dicts(), key=lambda x: x["iso639p3code"]
        )

        # Check file was written
        assert out_file.exists()
        read_back = pl.read_parquet(out_file)
        assert read_back.equals(result)
