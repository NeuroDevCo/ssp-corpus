"""Generate figures."""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import xarray as xr
from great_tables import GT, nanoplot_options
from matplotlib import pyplot as plt

from src import utils
from src.utils import IMG_PATH

COLOR = "grey"
COLOR2 = "lightgrey"
DPI = 1200
QUANT = [0.025, 0.5, 0.975]


def plot_ssp_examples():
    """Plot BLIF and LBIF as examples for manuscript."""

    word_ex = {"/blif/": [2, 6, 8, 3], "/lbif/": [6, 2, 8, 3]}
    title = ["/blif/ (SSP-following)", "/lbif/ (SSP-violating)"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    for idx, (w, si) in enumerate(word_ex.items()):
        ax = axes[idx]
        ax.plot(range(4), si, lw=3, c="k", zorder=-1)
        ax.scatter(range(4), si, s=300, zorder=0, ec="k", fc="w")
        l = w[1:-1]

        for i in range(4):
            ax.text(i, si[i], s=l[i].upper(), va="center", ha="center", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks(range(1, 10))

        ax.spines[["right", "top", "bottom"]].set_visible(False)
        ax.set_title(title[idx])

        if idx == 0:
            ax.set_ylabel("Sonority Index")
            ax.set_yticks(range(1, 10))
        else:
            ax.set_yticklabels([])
            ax.tick_params(length=0)
            ax.spines[["left"]].set_visible(False)

        ax.grid(True, alpha=0.25)
        ax.set_ylim(1, 10)

    fig.tight_layout()
    fig.set_size_inches(8, 3)
    fig.savefig(IMG_PATH / "ssp-examples.png", dpi=DPI)


def table_examples():
    """Table of SSP examples for manuscript."""
    langs = pl.read_parquet(utils.RESULTS_PATH / "languages.parquet")
    s = pl.read_parquet(utils.RESULTS_PATH / "syllables.parquet")
    w = pl.read_parquet(utils.RESULTS_PATH / "words.parquet")
    n_words = w.group_by("iso639p3code").len("n_words")
    codes = ["rus", "fin", "arz", "hin", "ell"]
    syllables = [
        "font",
        "fpɐɫ",
        "koht",
        "s̠trø̞m",
        "njäːɾ",
        "nisr",
        "st̪ɐ̃n",
        "pʰɛ",
        "ftis",
        "bɾos",
    ]
    df = (
        s.join(langs, on=list(set(langs.columns).intersection(s.columns)))
        .join(n_words, on="iso639p3code", how="left")
        .filter(pl.col("iso639p3code").is_in(codes), pl.col("syll").is_in(syllables))
        .sort(["is_ssp", "iso639p3code"])
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
        d.select(["iso639p3code", "phon", "word"])

        wlist.append(d)
    wlist = pl.concat(wlist)

    df = (
        df.join(wlist, on=["iso639p3code", "syll"])
        .with_columns(
            pl.col("sonority_2").list.join(" "),
            pl.col("sonority").list.join("-"),
            phon=pl.col("word") + pl.lit(" | /") + pl.col("phon") + pl.lit("/"),
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
                data_line_stroke_color="black",
                data_line_stroke_width=7,
                data_area_fill_color="orange",
                data_point_fill_color="white",
                data_point_stroke_color="black",
                data_point_stroke_width=7,
            ),
        )
        .opt_table_font(font=["Arial"])
    )
    tbl.gtsave(Path("img/examples.png"), zoom=10)

    return tbl


def plot_model_langs(
    data: pl.DataFrame, post: xr.DataArray | xr.Dataset, path: Path, xlim: float = 0.5
):
    """Main figure of the manuscript with model predictions."""
    ssp = data.group_by(["iso639p3code"]).agg(pl.col("is_ssp").mean())

    langs = 1 / (1 + (np.exp(-(post.α + post.u))))
    alpha = 1 / (1 + (np.exp(-(post.α)))).quantile(QUANT, dim="sample")

    hdi = langs.quantile(QUANT, dim="sample")
    n_obs = data.group_by("iso639p3code").len()

    fig, ax = plt.subplots(1, 1, figsize=(8, 9))
    ax.set_title("A", loc="left")
    ax.axvspan(
        alpha.sel(quantile=QUANT[0]),
        alpha.sel(quantile=QUANT[2]),
        color=COLOR2,
        alpha=0.85,
        lw=0,
        zorder=-1,
    )
    ax.axvline(alpha.sel(quantile=0.5), c="grey", zorder=-1, ls="dotted", lw=0.5)
    ax.text(
        alpha.sel(quantile=QUANT[0]) - 0.03,
        -2,
        s=f"Global median ({alpha.sel(quantile=QUANT[1]):.2%}, $n$ = {len(data):,})",
        ha="right",
        va="center",
        fontsize=7,
    )
    ax.hlines(
        -2,
        alpha.sel(quantile=QUANT[0]),
        alpha.sel(quantile=QUANT[-1]),
        colors=["k"],
        lw=1.5,
        zorder=-1,
    )
    ax.scatter(alpha.sel(quantile=QUANT[1]), -2, s=50, marker="d", c="k")

    for i, li in enumerate((-hdi).argsort().sel(quantile=0.5)):
        l = langs.j.values[li]
        h = hdi.sel(j=l)
        n = n_obs.filter(pl.col("iso639p3code") == l).get_column("len")
        d = ssp.filter(pl.col("iso639p3code") == l).get_column("is_ssp")
        lang_label = f"{utils.iso639_p3_to_name(l)} ({h.sel(quantile=QUANT[0]):.2%}, $n$ = {n[0]:,})"
        ax.text(
            h.sel(quantile=QUANT[0]) - 0.01,
            lang_label,  # ty:ignore[invalid-argument-type]
            s=lang_label,
            ha="right",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round", "ec": "white", "fc": "white", "alpha": 0.25},
        )
        ax.hlines(
            lang_label,
            h.sel(quantile=QUANT[0]),
            h.sel(quantile=QUANT[-1]),
            colors=["k"],
            lw=1.5,
            zorder=-1,
        )
        label = "Model-predicted" if i == 0 else None
        ax.scatter(h.sel(quantile=0.5), lang_label, s=30, ec="k", fc="w", label=label)
        label = "Observed" if i == 0 else None
        ax.scatter(d, lang_label, c="orange", s=50, marker="x", label=label)

    ax.set_yticks([])
    ax.set_xlim(xlim, 1.0)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.grid(True, lw=0.5)
    ax.set_xlabel("Probability of SSP-compliance ($\\theta_{ij}$)")
    ax.grid(which="major", axis="x", linestyle="dotted")
    ax.legend(frameon=False, loc="best", bbox_to_anchor=(0.12, 0.38, 0.5, 0.5))

    ins_ax = ax.inset_axes((0.1, 0.10, 0.3, 0.2))
    ins_ax.patch.set_alpha(0.5)
    plot_model_preds(post, data, ins_ax)

    ins_ax2 = ax.inset_axes((0.1, 0.5, 0.3, 0.2))
    ins_ax2.patch.set_alpha(0.5)
    plot_model_params(post, ins_ax2)

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=DPI)


def plot_model_params(post: xr.DataArray | xr.Dataset, ax: plt.Axes):
    """Plot model parameters."""
    post_delta = post.β
    y = 900
    hdi = post_delta.quantile(QUANT, dim="sample")

    label = (
        f"$\\beta$ = {hdi.values[1]:.3f}\n[{hdi.values[-0]:.3f}, {hdi.values[-1]:.3f}]"
    )
    ax.text(hdi.sel(quantile=QUANT[1]), y * 1.07, label, va="bottom", ha="center")
    ax.hlines(
        y * 1.02,
        hdi.sel(quantile=QUANT[0]),
        hdi.sel(quantile=QUANT[-1]),
        lw=1.5,
        colors=["k"],
        zorder=-1,
    )
    ax.scatter(hdi.sel(quantile=0.5), y * 1.02, zorder=0, s=40, ec="k", fc="w")
    ax.hist(post_delta, bins=15, edgecolor="w", linewidth=1, color="lightgrey")
    ax.axvline(0, lw=1, ls="dashed", c=COLOR)
    ax.set_ylim(0, y * 1.1)
    # ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.set_ylabel("Number of posterior draws")
    ax.set_xlabel(
        "Posterior distribution of regression\ncoefficient for syllabic frequency ($\\beta$)"
    )
    ax.set_title("B", loc="left")
    ax.spines[["left", "top", "right"]].set_visible(False)


def plot_model_preds(
    post: xr.DataArray | xr.Dataset,
    data: pl.DataFrame,
    ax: plt.Axes | None = None,
    path: Path | None = None,
):
    """Plot model predictions."""

    x_nd = np.linspace(1, 7, 100)
    x_raw = data["freq"].to_numpy()
    x = xr.DataArray((x_nd - x_raw.mean()) / x_raw.std(), dims=["x"])
    y = 1 / (1 + (np.exp(-(post.α + post.β * x))))
    hdi = y.quantile(QUANT, dim="sample")

    y_lang = 1 / (1 + (np.exp(-((post.α + post.u) + (post.β + post.w) * x))))
    hdi_lang = y_lang.quantile(0.5, dim="sample")

    ax_missing = ax is None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    ax.plot(x, hdi.sel(quantile=0.5), c="k", lw=2, zorder=0, label="Global median")

    for i, l in enumerate(hdi_lang.j.values):
        label = "Individual languages" if i == 0 else None
        ax.plot(
            x, hdi_lang.sel(j=l), alpha=0.5, color="lightgrey", label=label, zorder=-1
        )

    tick_labels = np.arange(1, 8)
    ticks = (tick_labels - x_raw.mean()) / x_raw.std()
    ax.set_xlabel("Lexical frequency (Zipf score)")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Syllabic frequency (Zipf score)")
    ax.set_ylabel("Probability of SSP-compliance ($\\theta_{ij}$)")
    # ax.grid(which="major", axis="both", linestyle="dotted")
    ax.set_axisbelow(True)
    ax.set_ylim(0.5, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("C", loc="left")

    if ax_missing:
        fig.set_size_inches(8, 6.5)

    if ax_missing and path is not None:
        fig.savefig(path, dpi=DPI)


def plot_model_random(post: xr.DataArray, ax: plt.Axes):
    """Plot model random effects."""
    u = 1 / (1 + (np.exp(-(post.α + post.u).mean("sample"))))
    w = (post.β + post.w).mean("sample") / 4

    ax.axhline(1 / (1 + (np.exp(-(post.α.mean("sample"))))), lw=2, c=COLOR2)
    ax.axvline(post.β.mean("sample") / 4, lw=2, c=COLOR2)
    ax.axvline(0, ls="dotted", c=COLOR)

    ax.scatter(x=w, y=u, color="orange", marker="x")

    for j in post.j.values:
        x, y = u.sel(j=j), w.sel(j=j)
        ax.annotate(
            j,
            xy=(x, y),
            xycoords="data",
            xytext=(0.36, 0.68),
            textcoords="axes fraction",
            arrowprops={"facecolor": "black", "shrink": 0.05},
            horizontalalignment="right",
            verticalalignment="top",
        )
    ax.set_ylabel("Probability of SSP compliance ($\\alpha_j$)")
    ax.set_xlabel("Frequency effect ($\\beta_j$)")
    ax.set_title("C) Random effect correlation")


def main(
    data_path: Path = utils.RESULTS_PATH / "syllables.parquet",
    post_path: Path = utils.RESULTS_PATH / "posterior.nc",
    img_path: Path = utils.IMG_PATH,
    **kwargs,
):
    """Main script."""
    data = pl.read_parquet(data_path)
    idata = az.from_netcdf(post_path, engine="netcdf4")
    post = az.convert_to_dataset(idata.posterior).stack(sample=("chain", "draw"))

    table_examples()
    plot_ssp_examples()
    plot_model_langs(data, post, path=img_path / "model-langs.png", **kwargs)


if __name__ == "__main__":
    main()

    noflat_path = utils.RESULTS_PATH / "no-flat"
    main(
        data_path=noflat_path / "syllables.parquet",
        post_path=noflat_path / "posterior.nc",
        img_path=utils.IMG_PATH / "no-flat",
        xlim=0.3,
    )
