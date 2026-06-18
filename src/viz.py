from collections import defaultdict
import polars as pl

from matplotlib import pyplot as plt
from matplotlib import colors, ticker
import numpy as np

from src import utils

COLOR = "black"
DPI = 800


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
        pl.read_parquet(utils.OUT_PATH / "languages.parquet")
        .unique("family_id")
        .select("family_id")
        .sort("family_id")
    )
    f = langs.get_column("family_id").sort().to_list()
    cm = plt.get_cmap(cmap)
    vals = cm(np.linspace(0, 1, len(f)), alpha=alpha)

    return {k: colors.rgb2hex(v, keep_alpha=True) for k, v in zip(f, vals)}


def plot_results():
    s = pl.read_parquet(utils.OUT_PATH / "syllables.parquet")
    langs = (
        pl.read_parquet(utils.OUT_PATH / "languages.parquet")
        .sort("family_id")
        .select(["iso639p3code", "family_id"])
    )

    data = (
        s.group_by("iso639p3code")
        .agg(pl.col("is_ssp").mean(), n=pl.len())
        .with_columns(
            se=((pl.col("is_ssp") * (1 - pl.col("is_ssp"))) / pl.col("n")).sqrt()
        )
    )

    p = (
        pl.read_parquet(utils.OUT_PATH / "posterior.parquet")
        .join(data, left_on="param", right_on="iso639p3code", how="inner")
        .join(langs, left_on="param", right_on="iso639p3code", how="inner")
        .with_columns(
            qtile=pl.col("median").qcut([0.5], labels=["Bottom half", "Top half"])
        )
        .sort(["qtile", "median"], descending=[False, True])
    )

    f = p.unique("family_id").get_column("family_id").sort().to_list()
    cm = plt.get_cmap("rainbow")
    vals = cm(np.linspace(0, 1, len(f)), alpha=1)
    clrs = {k: colors.rgb2hex(v, keep_alpha=True) for k, v in zip(f, vals)}
    p = p.partition_by("qtile", as_dict=True)

    counter = defaultdict(int)

    fig, axes = plt.subplots(1, len(p), figsize=(7, 5), sharex=True)
    for i, d in enumerate(p.values()):
        ax = axes[i]

        ax.hlines(
            y=d["param"],
            xmin=d["lower"],
            xmax=d["upper"],
            colors=[COLOR],
            linewidths=[1],
        )
        ax.scatter(d["median"], d["param"], c=COLOR, s=10)
        # ax.scatter(d["is_ssp"], d["param"], c="orange", s=20, marker="|", zorder=-1)

        ax.tick_params(axis="y", labelsize=6.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0%}"))
        ax.xaxis.grid(True, color="lightgrey", linestyle=":")

        ticks = ax.get_yticks()
        labels = [f"{p:.1%} ($N={n:5,d}$)" for p, n in zip(d["median"], d["n"])]
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(labels, ha="left")
        ax2.tick_params(axis="y", labelsize=5)

        for row in d.iter_rows(named=True):
            label = row["family_id"] if counter[row["family_id"]] == 0 else None

            ax.scatter(0, row["param"], c=clrs[row["family_id"]], s=5, label=label)

            counter[row["family_id"]] += 1
    fig.legend(
        title=r"$\bf{Language~family}$",
        ncol=len(counter) // 2,
        alignment="center",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.savefig(utils.IMG_PATH / "results.png", dpi=DPI, bbox_inches="tight")


if __name__ == "__main__":
    plot_results()
