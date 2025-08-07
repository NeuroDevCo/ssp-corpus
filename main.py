"""Main script."""

from pathlib import Path
import polars as pl
from src import phonology, figures, model, scraping


def main():
    """Main script."""
    langs = scraping.get_languages().rename({"iso639p3code": "lang"})
    words, syll = phonology.preprocess_languages()

    data = phonology.proportion_by(
        syll, by=["lang", "language_name", "family_name"]
    ).filter(pl.col("family_name") != "Bookkeeping")
    fit, standata = model.fit_model(data)
    posterior = model.summarize_posterior(fit, standata)

    figures.plot_map(words)
    figures.plot_proportions(data)
    figures.plot_estimates(data, posterior, file=Path("img", "estimates.png"))

    phonology.cast_to_str(words).write_csv(Path("out", "words.csv"))
    phonology.cast_to_str(syll).write_csv(Path("out", "syllables.csv"))
    data.write_csv(Path("out", "data.csv"))
    posterior.write_csv(Path("out", "posterior.csv"))

    return langs, words, syll, data, posterior


if __name__ == "__main__":
    LANGS, WORDS, SYLL, DATA, POST = main()
