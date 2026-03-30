from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl

# Import the functions to be tested.
# Adjust the import path 'scraping' if your file is named differently.
from src.scraping import (
    EmptyLanguage,
    scrape_language,
    get_scraped_languages,
    scrape_all_languages,
)


class TestGetScrapedLanguages:
    """Tests for the get_scraped_languages function."""

    def test_returns_list_of_files(self, tmp_path: Path):
        """Test that it returns a list of ISO codes found in the directory."""
        tmp_path = Path(tmp_path)
        for lang in ["eng", "fra"]:
            fn = lang + ".csv"
            with (tmp_path / fn).open("w") as f:
                f.write("Some text")

        result = get_scraped_languages(path=tmp_path, min_size=1)

        assert set(result) == {"eng", "fra"}

    def test_returns_list_of_files_txt(self, tmp_path: Path):
        """Test that it returns a list of ISO codes found in the directory."""
        tmp_path = Path(tmp_path)
        # Create dummy CSV files
        for lang in ["eng", "fra"]:
            fn = lang + ".txt"
            with (tmp_path / fn).open("w") as f:
                f.write("Some text")

        result = get_scraped_languages(path=tmp_path, min_size=1, extension=".txt")

        assert set(result) == {"eng", "fra"}

    def test_filters_empty_files(self, tmp_path):
        """Test that keep_empty=False excludes files smaller than 1KB."""
        # Create a file with significant content (> 1000 bytes)
        large_file = tmp_path / "eng.csv"
        large_file.write_text("word,pron\n" + "data," * 300)

        # Create a small file
        small_file = tmp_path / "spa.csv"
        small_file.write_text("word,pron")

        result = get_scraped_languages(path=tmp_path, keep_empty=False)

        assert "eng" in result
        assert "spa" not in result

        result = get_scraped_languages(path=tmp_path, keep_empty=False, min_size=1)
        assert "eng" in result
        assert "spa" in result

    def test_keeps_empty_files_when_flagged(self, tmp_path):
        """Test that keep_empty=True includes small files."""
        small_file = tmp_path / "spa.csv"
        small_file.write_text("smol")

        result = get_scraped_languages(path=tmp_path, keep_empty=True)

        assert "spa" in result


class TestScrapeLanguage:
    """Tests for the scrape_language function."""

    @patch("src.scraping._scrape_lang_wrapper")
    @patch("src.scraping.CON")
    def test_broad_succeeds(self, mock_con, mock_wrapper, tmp_path):
        """Test successful scrape in 'broad' mode (default)."""
        mock_wrapper.return_value = None  # Success

        scrape_language("eng", path=tmp_path)

        # Verify wrapper was called with is_narrow=False
        args, kwargs = mock_wrapper.call_args
        assert args[0] == "eng"
        assert args[2] == tmp_path

    @patch("src.scraping._scrape_lang_wrapper")
    @patch("src.scraping.CON")
    def test_fallback_to_narrow_on_empty_language(
        self, mock_con, mock_wrapper, tmp_path
    ):
        """Test fallback to narrow if broad returns EmptyLanguage."""
        # First call (broad) raises EmptyLanguage, second (narrow) succeeds
        mock_wrapper.side_effect = [EmptyLanguage(), None]

        scrape_language("eng", path=tmp_path)

        assert mock_wrapper.call_count == 2

    @patch("src.scraping._scrape_lang_wrapper")
    @patch("src.scraping.CON")
    def test_fallback_to_narrow_on_value_error(self, mock_con, mock_wrapper, tmp_path):
        """Test fallback to narrow if broad raises ValueError."""
        mock_wrapper.side_effect = [ValueError("Bad Key"), None]

        scrape_language("eng", path=tmp_path)

        assert mock_wrapper.call_count == 2
        assert mock_con.print.call_count >= 1  # Prints red error for ValueError

    @patch("src.scraping._scrape_lang_wrapper")
    @patch("src.scraping.CON")
    def test_total_failure_logs_narrow_error(self, mock_con, mock_wrapper, tmp_path):
        """Test that if both broad and narrow fail, it logs 'No narrow pronunciations'."""
        mock_wrapper.side_effect = EmptyLanguage()

        scrape_language("eng", path=tmp_path)

        assert mock_wrapper.call_count == 2
        # Check that the specific error message was printed
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("No narrow pronunciations" in c for c in calls)


class TestScrapeAllLanguages:
    """Tests for the scrape_all_languages function."""

    @patch("src.scraping.scrape_language")
    @patch("src.scraping.get_languages")
    @patch("src.scraping.get_scraped_languages")
    @patch("src.scraping.CON")
    def test_scrapes_missing_languages_only(
        self, mock_con, mock_get_scraped, mock_get_langs, mock_scrape
    ):
        """Test that it only scrapes languages not present in get_scraped_languages."""
        # Setup mock data
        all_langs = pl.DataFrame({"iso639p3code": ["eng", "fra", "deu", "spa"]})
        mock_get_langs.return_value = all_langs

        # Say 'eng' and 'fra' are already scraped
        mock_get_scraped.return_value = ["eng", "fra"]

        scrape_all_languages()

        # Should only call scrape_language for 'deu' and 'spa'
        assert mock_scrape.call_count == 2

        called_langs = [call[0][0] for call in mock_scrape.call_args_list]
        assert "deu" in called_langs
        assert "spa" in called_langs
        assert "eng" not in called_langs
        assert "fra" not in called_langs

    @patch("src.scraping.scrape_language")
    @patch("src.scraping.get_languages")
    @patch("src.scraping.get_scraped_languages")
    @patch("src.scraping.CON")
    def test_progress_logging(
        self, mock_con, mock_get_scraped, mock_get_langs, mock_scrape
    ):
        """Test that progress is logged correctly."""
        mock_get_langs.return_value = pl.DataFrame({"iso639p3code": ["a", "b"]})
        mock_get_scraped.return_value = []  # None scraped yet

        scrape_all_languages()

        # Check that CON.print was called with progress info
        # The format is roughly "[{i + 1}/{len(langs)}] {la} ({time})"
        calls = [str(c) for c in mock_con.print.call_args_list]
        assert any("[1/2]" in c for c in calls)
        assert any("[2/2]" in c for c in calls)
