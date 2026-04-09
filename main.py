"""Main script."""

from pathlib import Path

from matplotlib import pyplot as plt
import polars as pl

from src.languages import get_languages
from src.words import get_words
from src.syllables import get_syllables
from src.model import fit_model, get_stan_data, get_posterior

IMG_PATH: Path = Path("img")


def make_data():
    langs = get_languages()
    words = get_words(langs)
    sylls = get_syllables(words)

    return langs, words, sylls


if __name__ == "__main__":
    langs, words, sylls = make_data()

    sylls = sylls.drop_nulls(["freq"])

    stan_data = get_stan_data(sylls)
    stan_file = Path("model/model.stan")
    model, samples = fit_model(stan_data, stan_file)
    samples.save_csvfiles(Path("model", "samples"))

    get_posterior(samples).write_parquet(Path("out", "posterior.parquet"))
    post = get_posterior(samples, params="theta_j").sort("median", descending=True)
    post_pop = (
        get_posterior(samples)
        .filter(pl.col("param").is_in(["beta[1]", "beta[2]", "sigma_mu", "theta"]))
        .sort("median")
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    ax.hlines(y=post["param"], xmin=post["lower"], xmax=post["upper"])
    ax.scatter(post["median"], post["param"], s=10)
