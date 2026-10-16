"""Get pronunciations from Wiktionary."""

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import wikipron
from rich.console import Console
from rich.live import Live
from wikipron.config import _PHONEMES_REGEX, _PHONES_REGEX

from src import languages, utils

logger = logging.getLogger("pyppeteer")

EXCEPTIONS = (ValueError, KeyError, KeyboardInterrupt, FileNotFoundError)
type Error = ValueError | KeyError | KeyboardInterrupt | FileNotFoundError

CON: Console = Console()


@dataclass
class Config(wikipron.Config):
    language: str
    syllable_boundaries: bool = True
    casefold: bool = True
    stress: bool = True
    narrow: bool = True
    cut_off_date: str | None = None
    dialect: str | None = None
    segment: bool = True
    tone: bool = True
    skip_spaces_word: bool = True
    skip_spaces_pron: bool = True
    parens: str = "expand"

    def __post_init__(self):
        self.language = self._get_language(self.language)
        self.casefold: Callable[[str], str] = self._get_casefold(self.casefold)
        self.cut_off_date: str = self._get_cut_off_date(self.cut_off_date)
        self.extract_word_pron = self._get_extract_word_pron(self.language)
        self.pron_xpath_selector: str = self._get_pron_xpath_selector(
            self.language, self.dialect
        )
        self.ipa_regex: str = _PHONES_REGEX if self.narrow else _PHONEMES_REGEX
        self.process_pron = self._get_process_pron(
            self.stress, self.syllable_boundaries, self.segment, self.tone
        )
        self.restart_key = None


@dataclass
class ProgressStatus:
    lang: str
    lang_progress: tuple[int, int] | None = None
    timestamp: datetime | None = None
    i: int = 0
    mode: str = "narrow"
    mode_index: int = 0
    color: str = "black"
    status: str = "Init"
    exception: Error | None = None

    def render(self, time_format: str = "%Y-%m-%d %H:%M:%S") -> str:
        if self.timestamp is None:
            self.timestamp = datetime.now()

        t = datetime.strftime(self.timestamp, time_format)

        head = f"{self.lang:20}"
        if self.lang_progress:
            i, n = self.lang_progress
            head = f"({i}/{n}) " + head

        text = f"{head} | {self.mode:6} mode ({self.mode_index + 1}/2) | [{self.color}]{self.status}[/{self.color}] {t} | Found {self.i}"

        if self.exception:
            text += str(self.exception)

        return text

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()


def scrape_language(
    lang: str,
    path: Path = Path("data/raw"),
    delete_empty: bool = False,
    progress: tuple[int, int] | None = None,
) -> bool:
    """Get pronunciations for language.

    Args:
        lang (str): ISO-639-3 code of language.
        path (Path, optional): Path to the folder in which the file will be created and saved: Should empty CSV be deleted? Defaults to True.
        delete_empty (bool, optional): Should empty CSV be deleted? Defaults to True.
        **kwargs: Additional arguments passed to `wikipron.Config`.
    """
    path.mkdir(exist_ok=True, parents=True)
    fn = path / f"{lang}.tsv"

    label = f"{lang} ({utils.iso639_p3_to_name(lang)})"[:20]
    idx, i = 0, 0
    mode = "narrow"

    p = ProgressStatus(label, progress, datetime.now(UTC), i, mode, idx, "blue")
    c = Config(language=lang)

    with Live("", console=CON, refresh_per_second=10) as live:
        live.update(p.render())

        if fn.exists() and idx == 0:
            p.status = "Skip"
            p.color = "yellow"
            p.timestamp = datetime.now(UTC)
            live.update(p.render())

            return True

        for idx, mode in enumerate(["narrow", "broad"]):
            c.narrow = mode == "narrow"
            p.mode_index = idx
            p.mode = mode
            try:
                with fn.open("a", encoding="utf-8", newline="") as f:
                    f.write("phon\tword\n")

                    for word, pron in wikipron.scrape(c):
                        f.write(f"{word}\t{pron}\n")
                        i += 1
                        p.i = i
                        live.update(p.render())

            except EXCEPTIONS as e:
                p.status = "Error"
                p.color = "red"
                p.timestamp = datetime.now(UTC)
                p.exception = e
                live.update(p.render())

                if i == 0 and delete_empty:
                    fn.unlink()

                return False

        p.status = "Done"
        p.color = "green"
        p.timestamp = datetime.now(UTC)
        live.update(p.render())

        if i == 0 and delete_empty:
            fn.unlink()

    return True


def main(langs: str | list[str] | None = None, max_attempts: int = 10, **kwargs):
    """Scrape all languages."""
    if isinstance(langs, str):
        langs = [langs]

    if langs is None:
        langs = languages.get_languages().get_column("iso639p3code").to_list()

    for idx, lang in enumerate(langs):
        status = False
        i = 0
        while not status and i < max_attempts:
            status = scrape_language(lang, progress=(idx, len(langs)), **kwargs)
            i += 1
            if not status:
                time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Language scraper",
        description="Scrape a list of languages",
        epilog="Use Wikipron to scrape pronunciations from Wiktionary",
    )
    parser.add_argument("--language", "-l", default=None)
    args = parser.parse_args()
    main(args.language)
