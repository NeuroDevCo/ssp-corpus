"""Main script."""

from functools import lru_cache
from src import phonology, figures, model, scraping


@lru_cache
def main():
    figures.plot_map()
    langs = scraping.get_languages().rename({"iso639p3code": "lang"})
    words, syll = phonology.preprocess_langs()
    data = model.proportion_by(syll, by=["lang", "language_name", "family_name"])
    data.write_csv("out", "data.csv")
    results = fit_model(data, group_var="family_name")

    data_preds = data.join(
        langs.select(["lang", "n_phon", "n_vow", "n_cons"]), on="lang", how="left"
    ).drop_nulls()
    results_base = fit_model(data_preds, group_var="family_name")

    results_cons = fit_model_pred(data_preds, "n_cons", group_var="family_name")
    print(compare_models(results_base[2], results_cons[2]))

    results_phon = fit_model_pred(data_preds, "n_phon", group_var="family_name")
    print(compare_models(results_base[2], results_phon[2]))

    results_vow = fit_model_pred(data_preds, "n_vow", group_var="family_name")
    print(compare_models(results_base[2], results_vow[2]))

    figures.plot_proportions(data)
    return langs, words, syll, data, results


if __name__ == "__main__":
    LANGS, WORDS, SYLL, DATA, RESULTS = main()
