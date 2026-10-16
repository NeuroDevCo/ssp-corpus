"""Generate figures."""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import xarray as xr
from great_tables import GT, nanoplot_options
from iso639 import Language
from matplotlib import colors
from matplotlib import pyplot as plt

from src import languages, utils
from src.utils import IMG_PATH

COLOR = "grey"
COLOR2 = "lightgrey"
DPI = 1200
QUANT = [0.025, 0.5, 0.975]


def get_color_codes(
    x: list[str], cmap: str = "rainbow", alpha: float = 1.0
) -> dict[str, str]:
    """Assign color to each language family.

    Args:
        x (list[str]): Language families.
        cmap (str, optional): Color map to use. Defaults to "rainbow".
        alpha (float, optional): Opacity of the colors. Defaults to 1.0 (full opacity).

    Returns:
        dict: Language family and hex color pairs.
    """

    cm = plt.get_cmap(cmap)
    vals = cm(np.linspace(0, 1, len(x)), alpha=alpha)

    return {k: colors.rgb2hex(v, keep_alpha=True) for k, v in zip(x, vals)}


data_path: Path = utils.RESULTS_PATH / "syllables.parquet"
data = pl.read_parquet(data_path)
families = languages.get_languages().select(["iso639p3code", "family_id"])

families = {k: v for k, v in families.iter_rows()}
ssp = data.group_by(["iso639p3code"]).agg(pl.col("is_ssp").mean())

CLRS = get_color_codes(
    list({families[l if l != "fas" else "pes"] for l in ssp["iso639p3code"]}),
    cmap="tab10",
)
CLRS = {k: CLRS[v] for k, v in families.items() if k in ssp["iso639p3code"]}


def iso639_p3_to_name(p3: str):
    try:
        lang = Language.from_part3(p3)
        return str(lang.name)
    except (LookupError, ValueError, AttributeError):
        return


def plot_ssp_examples():
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
                data_area_fill_color="grey",
                data_point_fill_color="white",
                data_point_stroke_color=COLOR,
                data_point_stroke_width=7,
            ),
        )
    )
    tbl.gtsave(Path("img/examples.png"), zoom=10)

    return tbl


def plot_model_langs(data: pl.DataFrame, post: xr.DataArray, path: Path):
    ssp = data.group_by(["iso639p3code"]).agg(pl.col("is_ssp").mean())

    langs = 1 / (1 + (np.exp(-(post.α + post.u))))
    alpha = 1 / (1 + (np.exp(-(post.α)))).quantile(QUANT, dim="sample")

    hdi = langs.quantile(QUANT, dim="sample")
    n_obs = data.group_by("iso639p3code").len()
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    ax.axvspan(
        alpha.sel(quantile=QUANT[0]),
        alpha.sel(quantile=QUANT[-1]),
        color=COLOR2,
        alpha=0.4,
        lw=0,
        zorder=-1,
    )
    ax.axvspan(
        alpha.sel(quantile=QUANT[1]),
        alpha.sel(quantile=QUANT[-2]),
        color=COLOR2,
        alpha=0.85,
        lw=0,
        zorder=-1,
    )
    ax.axvline(alpha.sel(quantile=0.5), c="k", zorder=-1)
    for li in (-hdi).argsort().sel(quantile=0.5):
        l = langs.j.values[li]
        h = hdi.sel(j=l)
        n = n_obs.filter(pl.col("iso639p3code") == l).get_column("len")
        d = ssp.filter(pl.col("iso639p3code") == l).get_column("is_ssp")
        lang_label = f"{iso639_p3_to_name(l)} ($n$={n[0]:,})"
        ax.hlines(
            lang_label,
            h.sel(quantile=QUANT[1]),
            h.sel(quantile=QUANT[-2]),
            colors=["k"],
            lw=3.5,
            zorder=-1,
        )
        ax.hlines(
            lang_label,
            h.sel(quantile=QUANT[0]),
            h.sel(quantile=QUANT[-1]),
            colors=["k"],
            lw=2,
            zorder=-1,
        )

        ax.scatter(h.sel(quantile=0.5), lang_label, s=50, ec="k", fc="w")
        ax.scatter(d, lang_label, c="k", s=100, marker="x")

    ax.set_xlabel("Probability of SSP-compliance ($\\theta_{ij}$)")
    ax.grid(which="major", axis="x", linestyle="dotted")

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=DPI)


def plot_model_params(post: xr.DataArray, ax: plt.Axes):

    post_delta = post.β / 4
    y = 2100
    hdi = post_delta.quantile(QUANT)

    ax.hlines(
        y,
        hdi.sel(quantile=QUANT[1]),
        hdi.sel(quantile=QUANT[-2]),
        lw=15,
        colors=[COLOR2],
        alpha=0.5,
        zorder=-1,
    )
    ax.hlines(
        y,
        hdi.sel(quantile=QUANT[0]),
        hdi.sel(quantile=QUANT[-1]),
        lw=15,
        colors=[COLOR2],
        alpha=0.5,
        zorder=-1,
    )
    ax.scatter(hdi.sel(quantile=0.5), y, s=100, zorder=0, c=COLOR, marker="x")
    ax.hist(post_delta, bins=15, edgecolor="w", linewidth=1, color=COLOR)
    ax.axvline(0, lw=1, ls="dashed", c=COLOR)
    ax.set_ylim(0, y + (y * 0.1))
    ax.set_ylabel("Number of posterior draws")
    ax.set_axisbelow(True)
    ax.set_title("A) Frequency effect ($\\beta_1 / 4$)")
    ax.set_xlabel("Difference in probability of SSP compliance")
    ax.grid(which="both", axis="x", linestyle="dotted")


def plot_model_preds(
    post: xr.DataArray,
    data: pl.DataFrame,
    ax: plt.Axes | None = None,
    path: Path | None = None,
):

    x_nd = np.linspace(1, 9, 100)
    x_raw = data["freq"].to_numpy()
    x = xr.DataArray((x_nd - x_raw.mean()) / x_raw.std(), dims=["x"])
    y = 1 / (1 + (np.exp(-(post.α + post.β * x))))
    hdi = y.quantile(QUANT, dim="sample")

    y_lang = 1 / (1 + (np.exp(-((post.α + post.u) + (post.β + post.w) * x))))
    hdi_lang = y_lang.quantile(0.5, dim="sample")

    ax_missing = ax is None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    ax.plot(x, hdi.sel(quantile=0.5), c="k", lw=2, zorder=0)

    for l in hdi_lang.j.values:
        ax.plot(x, hdi_lang.sel(j=l), alpha=0.5, color=COLOR2, zorder=-1)

    tick_labels = np.arange(1, 9)
    ticks = (tick_labels - x_raw.mean()) / x_raw.std()
    ax.set_xlabel("Lexical frequency (Zipf score)")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Syllabic frequency (Zipf score)")
    ax.set_ylabel("Probability of SSP-compliance ($\\theta_{ij}$)")
    ax.grid(which="major", axis="both", linestyle="dotted")
    ax.set_axisbelow(True)

    fig.set_size_inches(8, 6.5)

    if ax_missing and path is not None:
        fig.savefig(path, dpi=DPI)


def plot_model_random(post: xr.DataArray, ax: plt.Axes):
    u = 1 / (1 + (np.exp(-(post.α + post.u).mean("sample"))))
    w = (post.β + post.w).mean("sample") / 4

    ax.axhline(1 / (1 + (np.exp(-(post.α.mean("sample"))))), lw=2, c=COLOR2)
    ax.axvline(post.β.mean("sample") / 4, lw=2, c=COLOR2)
    ax.axvline(0, ls="dotted", c=COLOR)

    ax.scatter(x=w, y=u, color=COLOR, marker="x")

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


def plot_model(data: pl.DataFrame, post: xr.DataArray, path: Path | None = None):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # plot_model_params(post, axes[0])  # plot β posterior distribution
    plot_model_preds(post, data, ax)  # plot predictions
    # plot_model_random(post, axes[2])

    fig.tight_layout()
    fig.set_size_inches(6, 6)
    if path:
        fig.savefig(path, dpi=DPI)


def main(
    data_path: Path = utils.RESULTS_PATH / "syllables.parquet",
    post_path: Path = utils.RESULTS_PATH / "post.nc",
    img_path: Path = utils.IMG_PATH,
):
    data = pl.read_parquet(data_path)
    idata = az.from_netcdf(post_path, engine="netcdf4")
    post = az.convert_to_dataset(idata.posterior).stack(sample=("chain", "draw"))

    # table_examples()
    plot_ssp_examples()
    plot_model_langs(data, post, path=img_path / "model-langs.png")
    plot_model_preds(post, data, path=img_path / "model-preds.png")
    # plot_model(
    #     file=utils.RESULTS_PATH / "no-flat" / "post-freq.nc",
    #     path=utils.IMG_PATH / "model-posterior-noflat.png",
    # )


if __name__ == "__main__":
    main()

    noflat_path = utils.RESULTS_PATH / "no-flat"
    main(
        data_path=noflat_path / "syllables.parquet",
        post_path=noflat_path / "post-freq.nc",
        img_path=utils.IMG_PATH / "no-flat",
    )
