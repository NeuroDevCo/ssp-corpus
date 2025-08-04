"""Main script."""

from pathlib import Path
import polars as pl
from src import phonology, figures, model, scraping

if __name__ == "__main__":
    figures.plot_map()
    LANGS = scraping.get_languages().rename({"iso639p3code": "lang"})
    WORDS, SYLL = phonology.get_corpora()
    DATA = SYLL.join(
        LANGS.select(pl.col(["lang", "family_name", "language_name"])), on="lang"
    )
    Y, N, GROUPS = model.prepare_data(DATA, group_var="lang")
    RESULTS = model.fit_model(y=Y, n=N, groups=GROUPS)
    figures.plot_estimates_lang(
        RESULTS[1],
        GROUPS,
        file=Path("img", "estimates-lang.png"),
    )
    Y, N, GROUPS = model.prepare_data(DATA, group_var="family_name")
    FAM_RESULTS = model.fit_model(y=Y, n=N, groups=GROUPS)
    figures.plot_estimates(
        FAM_RESULTS[1],
        GROUPS,
        group_var="family_name",
        figsize=(8, 6),
        file=Path("img", "estimates-family.png"),
    )
