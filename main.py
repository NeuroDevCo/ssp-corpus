"""Main script."""

from pathlib import Path
import polars as pl
from src import phonology, figures, model, scraping


def main():
    langs = scraping.get_languages().rename({"iso639p3code": "lang"})
    words, syll = phonology.preprocess_langs()
    data = model.proportion_by(syll, by=["lang", "language_name", "family_name"])
    data.write_csv(Path("out", "data.csv"))

    data_preds = data.join(
        langs.select(["lang", "n_phon", "n_vow", "n_cons"]), on="lang", how="left"
    ).drop_nulls()

    models = {
        "results_base": model.fit_model(data_preds),
        "results_phon": model.fit_model(data_preds, pred_var="n_phon"),
        "results_cons": model.fit_model(data_preds, pred_var="n_cons"),
        "results_vow": model.fit_model(data_preds, pred_var="n_vow"),
    }
    comp = pl.from_pandas(model.compare_models(models)).write_csv(
        Path("out", "comparisons.csv")
    )
    comp.write_csv()
    figures.plot_proportions(data)
    figures.plot_map()

    return langs, words, syll, data, models, comp


if __name__ == "__main__":
    LANGS, WORDS, SYLL, DATA, RESULTS, COMP = main()
