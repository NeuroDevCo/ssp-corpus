"""Create figures."""

from pathlib import Path
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as cm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap
import arviz as az
from src import scraping, model


def get_color_codes(
    x: list[str] | np.ndarray,
    name: str = "family_name",
    cmap: str = "rainbow",
    alpha: float = 1.0,
) -> dict[str, str]:
    """Assign color to each language family.

    Args:
        x (list[str]): Language families.
        cmap (str, optional): Color map to use. Defaults to "rainbow".
        alpha (float, optional): Opacity of the colors. Defaults to 1.0 (full opacity).

    Returns:
        dict: Language family and hex color pairs.
    """
    f = sorted(list(set(x)))
    cmap = plt.get_cmap(cmap)
    vals = cmap(np.linspace(0, 1, len(f)), alpha=alpha)
    return {k: cm.rgb2hex(v, keep_alpha=True) for k, v in zip(f, vals)}


def plot_structures(
    datapath: str | Path = Path("out", "syllables.csv"),
    file: Path | str = Path("img", "structures.png"),
):
    df = pl.read_csv(datapath).join(
        scraping.get_languages().rename({"iso639p3code": "lang"}), on="lang"
    )
    fn = "family_name"
    _, axes = plt.subplots(nrows=len(df.unique(fn)))
    for fam, ax in zip(df.get_column(fn).unique().to_numpy(), axes):
        d = df.filter(pl.col(fn) == fam).sort("n", descending=True)
        ax.scatter(d.get_column("struct"), d.get_column("n"), s=1)
    if file:
        plt.savefig(file)
    return None


def plot_n_words(
    datapath: str | Path = Path("out", "words.csv"),
    file: str | Path = Path("img", "map.png"),
):
    """Plot the distribution of distinct word-forms extracted for each language."""


def plot_estimates_lang(
    idata: az.InferenceData,
    groups: list[str],
    file: Path | str = Path("img", "estimates-lang.png"),
):
    """Plot model estimations for each language."""
    data = (
        pl.from_pandas(az.summary(idata, hdi_prob=0.89).reset_index())
        .with_columns(
            pl.col("index").str.replace_all("\\[|alpha|\\]", ""),
            pl.col(pl.Float64).map_elements(model.invlogit, return_dtype=pl.Float64),
        )
        .filter(pl.col("index").is_in(groups))
        .join(
            scraping.get_languages().rename({"iso639p3code": "index"}),
            on="index",
        )
        .sort("mean", descending=True)
    )
    colors = get_color_codes(data["index"])
    fig, axes = plt.subplots(ncols=2, figsize=(13, 10))
    data2 = data.slice(0, len(data) // 2)
    data1 = data.slice(len(data2), len(data) - len(data2))
    for ax, d in zip(axes, (data1, data2)):
        for i, m, lo, hi in zip(d["index"], d["mean"], d["hdi_5.5%"], d["hdi_94.5%"]):
            ax.errorbar(m, i, xerr=(hi - lo) / 2, lw=2, c=colors[i], fmt="none")
            ax.scatter(m, i, marker="d", c=colors[i])
            ax.scatter(0.51, i, c=colors[i])
            label = f"{m:.1%} [{lo:.1%}, {hi:.1%}]"
            ax.text(0.53, i, s=label, fontsize=4.5, backgroundcolor="#ffffff80")
        texts = {
            f["index"]: f["language_name"]
            for f in d.select(["index", "language_name"]).to_dicts()
        }
        ax.set_yticklabels(labels=[texts[t.get_text()] for t in ax.get_yticklabels()])
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_xlim(0.5, 1.0)
        ax.grid(True, axis="x", ls=":")
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=1e3, bbox_inches="tight")


def plot_estimates(
    idata: az.InferenceData,
    groups: list[str],
    group_var: str = "lang",
    file: Path | str = Path("img", "estimates.png"),
    **kwargs,
):
    cols_keep = ["index", "family_name", "language_name"]
    if group_var in cols_keep:
        cols_keep.remove(group_var)
    data = (
        pl.from_pandas(az.summary(idata, hdi_prob=0.89).reset_index())
        .with_columns(
            pl.col("index").str.replace_all("\\[|alpha|\\]", ""),
            pl.col(pl.Float64).map_elements(model.invlogit, return_dtype=pl.Float64),
        )
        .filter(pl.col("index").is_in(groups))
        .join(
            scraping.get_languages()
            .rename({"iso639p3code": "lang"})
            .rename({group_var: "index"})
            .select(pl.col(cols_keep)),
            on="index",
        )
        .sort("mean", descending=True)
    )
    var = "family_name" if group_var == "lang" else "index"
    colors = get_color_codes(data[var])
    fig, ax = plt.subplots(ncol=2 if var == "lang" else 1, **kwargs)
    ax.errorbar(
        x=data["mean"],
        y=data["index"],
        xerr=(data["hdi_94.5%"] - data["hdi_5.5%"]) / 2,
        lw=2,
        c="#333232",
        fmt="none",
    )
    ax.scatter(data["mean"], data["index"], c="#333232", marker="d")
    ax.scatter(
        np.repeat(0.51, len(data)),
        data["index"],
        c=[colors[f] for f in data[var]],
    )
    texts = {f["index"]: f[var] for f in data.select(cols_keep).to_dicts()}
    ax.set_yticklabels(labels=[texts[t.get_text()] for t in ax.get_yticklabels()])
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, axis="x", ls=":")
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=1e3, bbox_inches="tight")


def plot_map():
    df = (
        pl.read_csv(Path("out", "words.csv"))
        .group_by(pl.col("lang"))
        .agg(pl.len().alias("n_pron"))
        .join(scraping.get_languages().rename({"iso639p3code": "lang"}), on="lang")
        .filter(pl.col("pop_size") > 0)
        .with_columns(
            pl.col("pop_size").sqrt().fill_nan(None).alias("exp_pop_size"),
            pl.col("n_pron").sqrt().fill_nan(None).alias("exp_n_pron"),
        )
        .drop_nulls(pl.col("exp_pop_size"))
        .sort("family_name")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    m = Basemap(
        projection="kav7",
        lon_0=0,
        resolution="c",
        ax=ax,
    )
    m.drawmapboundary(fill_color="white", linewidth=0)
    m.drawcoastlines(color="#cccccc")
    m.fillcontinents(color="#cccccc")
    colors = get_color_codes(df["family_name"], alpha=1 / 2)

    # plot populations
    for fam, col in colors.items():
        d = df.filter(pl.col("family_name") == fam)
        x, y = m(d["longitude"], d["latitude"])
        m.scatter(
            x,
            y,
            marker="o",
            s=(d["pop_size"] / 1e3).sqrt(),
            c=col,
            zorder=4,
            lw=0,
            label=fam,
            ax=ax,
        )
    legend_fam = ax.legend(
        markerscale=0.5,
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
        alignment="left",
        title="Language family",
        frameon=False,
    )
    x, y = m(df["longitude"], df["latitude"])
    pop_scatter = m.scatter(
        x,
        y,
        marker="o",
        s=(df["pop_size"] / 1e3).sqrt(),
        c=col,
        zorder=3,
        lw=0,
        alpha=0,
        label=fam,
        ax=ax,
    )
    for lg in legend_fam.legend_handles:
        lg.set_sizes([200])
        lg.set_alpha([1])
    handles, labels = pop_scatter.legend_elements(
        prop="sizes",
        alpha=0.6,
        num=5,
        func=lambda x: (x**2) / 1e3,
        mfc="#333232",
        mew=0,
    )
    legend_pop = ax.legend(
        handles,
        labels,
        loc="upper left",
        title="Population size (millions)",
        frameon=False,
        alignment="left",
        ncol=5,
        bbox_to_anchor=(0, 1.15),
    )
    legend_pop._legend_box.sep = 20

    # pot n pronunciations
    scatter_pron = m.scatter(
        x,
        y,
        marker="o",
        sizes=df["n_pron"].sqrt(),
        fc="none",
        ec="#333232",
        lw=1 / 2,
        zorder=4,
    )
    handles, labels = scatter_pron.legend_elements(
        prop="sizes",
        alpha=0.6,
        num=5,
        func=lambda x: (x**2),
        mec="#333232",
        fmt=FuncFormatter(lambda x, p: format(int(x), ",")),
        mfc="none",
    )
    legend_pron = ax.legend(
        handles,
        labels,
        loc="lower left",
        title="Number pronunciations retrieved from Wiktionary",
        ncol=5,
        alignment="left",
        frameon=False,
        bbox_to_anchor=(0, -0.15),
    )
    legend_pron._legend_box.sep = 10
    m.scatter(
        x,
        y,
        marker="o",
        s=1/2,
        c="#333232",
        zorder=5,
    )
    fig.add_artist(legend_pop)
    fig.add_artist(legend_fam)
    fig.add_artist(legend_pron)
    fig.tight_layout(pad=1, w_pad=1)
    fig.savefig(Path("img", "map.png"), dpi=1e3, bbox_inches="tight")


if __name__ == "__main__":
    pass
