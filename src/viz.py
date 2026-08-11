"""Generate figures."""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import xarray as xr
from great_tables import GT, nanoplot_options
from matplotlib import colors
from matplotlib import pyplot as plt

from src import utils
from src.utils import IMG_PATH

az.rcParams["data.load"] = True

COLOR = "darkslategrey"
DPI = 1200


def plot_ssp_examples():
    word_ex = {"/kɹɪsp/": [1, 7, 8, 3, 1], "/snæps/": [3, 5, 9, 1, 3]}
    colors = ["darkslategrey", "orange"]
    legend = ["/kɹɪsp/ (SSP-following)", "/snæps/ (SSP-violating)"]
    fig, ax = plt.subplots(1, 1)

    for idx, (_, si) in enumerate(word_ex.items()):
        ax.plot(range(5), si, lw=5, c=colors[idx], zorder=-1)
        ax.scatter(range(5), si, c=colors[idx], label=legend[idx], s=300, zorder=0)

    for idx in range(5):
        ax.text(
            idx,
            word_ex["/kɹɪsp/"][idx],
            s="kɹɪsp"[idx],
            color="white",
            va="center",
            ha="center",
            fontsize=10,
        )
        ax.text(
            idx,
            word_ex["/snæps/"][idx],
            s="snæps"[idx],
            va="center",
            ha="center",
            fontsize=10,
        )

    ax.spines[["right", "top", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_yticks(range(11))
    ax.set_ylabel("Sonority Index")

    fig.tight_layout()
    fig.set_size_inches(6, 2.5)
    fig.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(IMG_PATH / "ssp-examples.png", dpi=DPI)


def get_color_codes(cmap: str = "rainbow", alpha: float = 1.0) -> dict[str, str]:
    """Assign color to each language family.

    Args:
        x (list[str]): Language families.
        cmap (str, optional): Color map to use. Defaults to "rainbow".
        alpha (float, optional): Opacity of the colors. Defaults to 1.0 (full opacity).

    Returns:
        dict: Language family and hex color pairs.
    """
    langs = (
        pl.read_parquet(utils.RESULTS_PATH / "languages.parquet")
        .unique("family_id")
        .select("family_id")
        .sort("family_id")
    )
    f = langs.get_column("family_id").sort().to_list()
    cm = plt.get_cmap(cmap)
    vals = cm(np.linspace(0, 1, len(f)), alpha=alpha)

    return {k: colors.rgb2hex(v, keep_alpha=True) for k, v in zip(f, vals)}


def table_examples():
    langs = pl.read_parquet(utils.RESULTS_PATH / "languages.parquet")
    s = pl.read_parquet(utils.RESULTS_PATH / "syllables.parquet")
    w = pl.read_parquet(utils.RESULTS_PATH / "words.parquet")
    n_words = w.group_by("iso639p3code").len("n_words")

    s = s.join(langs, on=list(set(langs.columns).intersection(s.columns))).join(
        n_words, on="iso639p3code", how="left"
    )
    examples = s.filter(pl.col("syll").str.len_chars() > 2).partition_by("is_ssp")

    df = []
    for d in examples:
        df.append(d.unique(["iso639p3code"]).sample(5))

    df = (
        pl.concat(df)
        .sort("iso639p3code")
        .with_columns(
            pl.col("is_ssp").replace_strict(
                {True: "SSP-following", False: "SSP-violating"}
            ),
            pl.col("sonority"),
        )
    )

    wlist = []
    for r in df.iter_rows(named=True):
        d = w.filter(
            pl.col("syll").list.contains(r["syll"]),
            pl.col("iso639p3code") == r["iso639p3code"],
        ).select(["iso639p3code", "phon", "word"])
        d = d.sample(1)
        d.insert_column(-1, pl.Series("syll", [r["syll"]]))
        d.insert_column(-1, pl.Series("sonority_2", [r["sonority"]]))

        wlist.append(d)
    wlist = pl.concat(wlist)

    df = (
        df.join(wlist, on=["iso639p3code", "syll"])
        .with_columns(
            pl.col("sonority_2").list.join(" "),
            pl.col("sonority").list.join("-"),
            phon=pl.col("word") + pl.lit(" /") + pl.col("phon") + pl.lit("/"),
            syll=pl.lit(" /") + pl.col("syll") + pl.lit("/"),
        )
        .sort("is_ssp")
        .select(["is_ssp", "lang", "syll", "phon", "sonority", "sonority_2"])
    )
    tbl = (
        GT(df, groupname_col="is_ssp")
        .cols_label(
            lang="Language",
            syll="Syllable",
            phon="Original word",
            sonority="Sonority index",
            sonority_2="",
        )
        .fmt_nanoplot(
            columns="sonority_2",
            options=nanoplot_options(
                data_line_stroke_color=COLOR,
                data_line_stroke_width=7,
                data_area_fill_color="orange",
                data_point_fill_color="white",
                data_point_stroke_color=COLOR,
                data_point_stroke_width=7,
            ),
        )
    )
    tbl.gtsave(Path("img/examples.png"), zoom=10)

    return tbl


def plot_model(file: Path = utils.RESULTS_PATH / "post-freq.nc"):
    data = pl.read_parquet(utils.RESULTS_PATH / "syllables.parquet")
    idata = az.from_netcdf(file, engine="netcdf4")
    post = az.convert_to_dataset(idata.posterior).stack(sample=("chain", "draw"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(post.β, bins=50, edgecolor="w", linewidth=1, color="orange")
    axes[0].axvline(0, lw=1, ls="dashed", c="k")

    # plot predictions
    x_nd = np.linspace(0, 7, 100)
    x_raw = data["freq"].to_numpy()
    x = xr.DataArray((x_nd - x_raw.mean()) / x_raw.std(), dims=["x"])
    y = 1 / (1 + (np.exp(-(post.α + post.β * x))))
    hdi = y.quantile([0.025, 0.15, 0.5, 0.85, 0.975], dim="sample")

    axes[1].fill_between(
        x,
        hdi.sel(quantile=0.025),
        hdi.sel(quantile=0.975),
        alpha=0.5,
        color="orange",
        edgecolor="w",
    )
    axes[1].fill_between(
        x,
        hdi.sel(quantile=0.15),
        hdi.sel(quantile=0.85),
        alpha=0.5,
        color="orange",
        edgecolor="w",
    )
    axes[1].plot(x, hdi.sel(quantile=0.5), c="k", lw=2)

    # plot languages
    langs = 1 / (1 + (np.exp(-(post.α + post.u))))
    hdi = langs.quantile([0.025, 0.5, 0.975], dim="sample")

    axes[2].vlines(
        langs.j.values, hdi.sel(quantile=0.025), hdi.sel(quantile=0.975), colors=["k"]
    )
    for li in hdi.argsort().sel(quantile=0.5):
        l = langs.j.values[li]
        axes[2].scatter(l, hdi.sel(quantile=0.5, j=l), c="k")

    fig.savefig(utils.IMG_PATH / "model-posterior.png", dpi=DPI)


def main():
    table_examples()
    plot_ssp_examples()
