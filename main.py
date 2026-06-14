"""Main script."""

from src import languages, model, syllables, words


def main():
    langs = languages.main()
    w = words.main()
    s = syllables.main()
    p = model.main()

    return langs, w, s, p


if __name__ == "__main__":
    langs, w, s, p = main()
