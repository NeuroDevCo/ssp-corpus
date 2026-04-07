"""Main script."""

from pathlib import Path


from src.languages import get_languages
from src.words import get_words
from src.syllables import get_syllables

IMG_PATH: Path = Path("img")


def make_data():
    langs = get_languages()
    words = get_words(langs)
    sylls = get_syllables(words)

    return langs, words, sylls


if __name__ == "__main__":
    langs, words, sylls = make_data()
