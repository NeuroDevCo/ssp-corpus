"""Main script."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import polars as pl
from wordfreq import zipf_to_freq

from src.languages import get_languages
from src.words import get_words
from src.syllables import get_syllables
from src.model import fit_model, get_stan_data, get_posterior, Conf, get_epreds

IMG_PATH: Path = Path("img")
STAN_FILE: Path = Path("model/model.stan")


def make_data():
    langs = get_languages()
    words = get_words(langs)
    sylls = get_syllables(words)

    return langs, words, sylls


def make_model(sylls: pl.DataFrame, conf: Conf = Conf()):
    stan_data = get_stan_data(sylls)
    _, samples = fit_model(stan_data, STAN_FILE, conf=conf)
    post = get_posterior(samples)
    post.write_parquet(Path("out", "posterior.parquet"))

    # generate predictions
    x_new = np.linspace(np.min(stan_data["x"]), np.max(stan_data["x"]), 100)
    freq_new = x_new + np.median(stan_data["freq"])
    freq_p = zipf_to_freq(freq_new)
    new_data = {
        "N": len(x_new) * stan_data["J"],
        "J": stan_data["J"],
        "x": x_new,
        "j": np.arange(1, stan_data["J"] + 1),
    }

    preds = get_epreds(samples, new_data, scale="prob")
    strat_mat = np.expand_dims(freq_p, (0, 2))
    preds_strat = preds * strat_mat

    return post, preds_strat, preds, samples


if __name__ == "__main__":
    langs, words, sylls = make_data()
    post, pred, samples = make_model(sylls.sample(2_000))

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.mean(preds, axis=2).T)
    axes[1].plot(np.mean(preds_strat, axis=2).T)