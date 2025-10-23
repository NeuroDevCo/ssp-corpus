"""Create figures."""

from pathlib import Path
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as cm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap
from src import scraping, model


def get_color_codes(cmap: str = "rainbow", alpha: float = 1.0) -> dict[str, str]:
    """Assign color to each language family.

    Args:
        x (list[str]): Language families.
        cmap (str, optional): Color map to use. Defaults to "rainbow".
        alpha (float, optional): Opacity of the colors. Defaults to 1.0 (full opacity).

    Returns:
        dict: Language family and hex color pairs.
    """
    lang = (
        scraping.get_languages()
        .unique("family_name")
        .select("family_name")
        .sort("family_name")
    )
    f = lang.get_column("family_name").sort().to_list()
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


def plot_proportions(
    data: pl.DataFrame,
    file: Path | str = Path("img", "proportions.png"),
):
    """Plot model estimations for each language."""

    d = data.sort("prop", descending=True)
    d2 = d.slice(0, len(data) // 2)
    d1 = d.slice(len(d2), len(d) - len(d2))

    colors = get_color_codes()
    fig, axes = plt.subplots(ncols=2, figsize=(13, 10))
    for ax, d in zip(axes, (d1, d2)):
        for di in d.to_dicts():
            ax.scatter(0.41, di["lang"], c=colors[di["family_name"]])
            ax.scatter(di["prop"], di["lang"], marker="o", s=15, c="k")
            label = f"{di['prop']:.2%}"
            ax.text(
                0.43, di["lang"], s=label, fontsize=4.5, backgroundcolor="#ffffff80"
            )
        texts = {f["lang"]: f["language_name"] for f in d.to_dicts()}
        ax.set_yticklabels(labels=[texts[t.get_text()] for t in ax.get_yticklabels()])
        ax.yaxis.set_tick_params(labelsize=7)
        ax.set_xlim(0.4, 1.0)
        ax.grid(True, axis="x", ls=":")
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=1e3, bbox_inches="tight")


def plot_estimates(
    data: pl.DataFrame,
    posterior: pl.DataFrame,
    file: Path | str = Path("img", "estimates.png"),
):
    p = posterior.sort("mean", descending=True)
    colors = get_color_codes()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        x=p["mean"],
        y=p["param"],
        xerr=(p["upper"] - p["lower"]) / 2,
        lw=2,
        c="#333232",
        fmt="none",
    )
    ax.scatter(p["mean"], p["param"], marker="o", c="#333232", s=15)
    if data is not None:
        for f in p["param"]:
            prop = data.filter(pl.col("family_name") == f)
            ax.scatter(
                prop["prop"],
                prop["family_name"],
                marker="|",
                c=colors[f],
                alpha=0.5,
                zorder=1,
            )
    ax.scatter(np.repeat(0.39, len(p)), p["param"], c=[colors[f] for f in p["param"]])
    for d in p.to_dicts():
        label = f"{d['mean']:.1%} [{d['lower']:.1%}, {d['upper']:.1%}]"
        ax.text(0.41, d["param"], s=label, fontsize=4.5, backgroundcolor="#ffffff80")

    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_xlim(0.38, 1.0)
    ax.grid(True, axis="x", ls=":")
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=1e3, bbox_inches="tight")


def plot_map(data: pl.DataFrame):
    """Plot geographical distribution of scraped languages."""

    df = (
        data.group_by(pl.col("lang"))
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
    colors = get_color_codes(alpha=1 / 2)

    # plot populations
    for fam, col in colors.items():
        d = df.filter(pl.col("family_name") == fam)
        if len(d) > 0:
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
        s=1 / 2,
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
