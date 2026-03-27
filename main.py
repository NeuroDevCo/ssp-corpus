"""Main script."""

from jinja2.nodes import With

from pathlib import Path

import arviz as az
import polars as pl
from matplotlib import pyplot as plt

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
        models[m] = (model, data, samples, idata)

    return models


if __name__ == "__main__":
    langs, words, sylls = make_data()
    models = make_models(langs, words, sylls)

    # model, mdata, samples, idata = models["model"]

    # lang_lookup = mdata.unique("iso639p3code")
    # lang_lookup = lang_lookup.insert_column(
    #     0, pl.Series("param", range(len(lang_lookup))).cast(pl.String)
    # ).select(["param", "iso639p3code"])

    # summary = az.summary(idata, hdi_prob=0.95, var_names=["theta", "alpha", "sigma"])
    # summary.index.name = "param"
    # pop_params = (
    #     pl.from_pandas(summary, include_index=True)
    #     .rename({"hdi_2.5%": "lower", "hdi_97.5%": "upper"})
    #     .join(lang_lookup, on="param", how="left")
    #     .select(["param", "mean", "lower", "upper"])
    # )

    # summary = az.summary(idata, hdi_prob=0.95, var_names=["theta_j"])
    # summary.index.name = "param"
    # group_params = (
    #     pl.from_pandas(summary, include_index=True)
    #     .rename({"hdi_2.5%": "lower", "hdi_97.5%": "upper"})
    #     .with_columns(pl.col("param").str.replace_all("theta_j\\[|\\]", ""))
    #     .join(lang_lookup, on="param", how="left")
    #     .select(["iso639p3code", "mean", "lower", "upper"])
    #     .sort("mean", descending=True)
    # )

    # fig, axis = plt.subplots(1, 1, figsize=(5, 20))
    # for lang, mean, lower, upper in group_params.iter_rows(named=False):
    #     axis.errorbar(mean, lang, color="k", xerr=([mean - lower], [upper - mean]))
    # axis.scatter(group_params["mean"], group_params["iso639p3code"], color="k")
    # fig.savefig(Path("test.png"), dpi=1500)

    # data = pl.read_parquet(Path("out", "data.parquet"))

    # ax = plt.subplot()
    # ax.plot(
    #     [0, max(data["n"])],
    #     [0, max(data["n"])],
    #     color="grey",
    #     linewidth=1,
    #     linestyle="dotted",
    # )
    # ax.plot(
    #     [0, max(data["n"]) * 0.90],
    #     [0, max(data["n"]) * 0.90],
    #     color="tab:orange",
    #     linewidth=1,
    #     linestyle="dotted",
    # )
    # for la, n, p, s, n in data.iter_rows(named=False):
    #     ax.text(n, s, s=la, ha="center", va="top", fontsize=5)
    # # for la, n, p, s, n in data.iter_rows(named=False):
    # #     ax.text(n, s, s=la, ha="center", va="top", fontsize=5)
    # ax.set_xlabel("Number of syllables (rank)")
    # ax.set_ylabel("Number of SSP-following syllables")
    # plt.show()
    # fig.tight_layout()
    # fig.savefig(Path("img", "nwords_prop.png"), dpi=1_000)
