"""Main script."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

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


def make_model(data: dict, conf: Conf = Conf()):
    _, samples = fit_model(data, STAN_FILE, conf=conf)
    post = get_posterior(samples)
    post.write_parquet(Path("out", "posterior.parquet"))

    return samples, post


def make_preds(data: dict, x_new: np.ndarray, **kwargs):
    # generate predictions
    x_new_c = x_new - np.median(data["freq"])
    new_data = {
        "N": len(x_new) * data["J"],
        "J": data["J"],
        "x": x_new_c,
        "j": np.arange(1, data["J"] + 1),
    }

    preds = get_epreds(samples, new_data, scale="prob", **kwargs)

    return preds


if __name__ == "__main__":
    langs, words, sylls = make_data()
    data = get_stan_data(sylls.sample(1_000))

    samples, post = make_model(data)

    x_new = np.arange(3, 8)
    preds, preds_mrp = make_preds(data, x_new, group_adj=False)

    preds_j, preds_mrp_j = make_preds(data, x_new, group_adj=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(x_new, np.mean(preds_j, axis=2).T, alpha=1 / 5, color="lightgrey")
    axes[0].plot(x_new, np.median(preds, axis=2)[0], color="k")
    axes[0].scatter(x_new, np.median(preds, axis=2)[0], c="k")

    axes[1].plot(x_new, np.mean(preds_mrp_j, axis=2).T, alpha=1 / 5, c="lightgrey")
    axes[1].plot(x_new, np.median(preds_mrp, axis=2)[0], c="k")
    axes[1].scatter(x_new, np.median(preds_mrp, axis=2)[0], c="k", zorder=-1)

    for ax in axes:
        ax.set_ylim(0, 1)
