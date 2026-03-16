"""Main script."""

from pathlib import Path

import arviz as az
import arviz_plots as azp

from src.languages import get_languages
from src.words import get_words
from src.syllables import get_syllables
from src.model import get_data, get_stan_data, fit_model

IMG_PATH: Path = Path("img")


def make_data():
    langs = get_languages()
    words = get_words(langs)
    sylls = get_syllables(words)

    return langs, words, sylls


def make_models(langs, words, sylls):
    models = {}
    for m in ["model"]:
        data = get_data(langs, words, sylls, file=Path("out", "data.parquet"))
        stan_data = get_stan_data(data)
        stan_file = Path(f"model/{m}.stan")
        model, samples, idata = fit_model(stan_data, stan_file)
        models[m] = (model, samples, idata)

    return models


if __name__ == "__main__":
    langs, words, sylls = make_data()
    models = make_models(langs, words, sylls)

    model, samples, idata = models["model"]
    azp.plot_trace_dist(idata, coords={"a_dim_0": 1})
    summary = samples.summary()
    summary.filter(regex="theta", axis="index")

    az.plot_forest(
        idata, var_names="theta", figsize=(6, 32), r_hat=True, combined=True, textsize=8
    )
