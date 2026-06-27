"""Generate figures."""

from collections import defaultdict
from pathlib import Path
import polars as pl

from matplotlib import pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import seaborn as sns

from src import languages, model, utils

COLOR = "darkslategrey"
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


def plot_ssp_langs():
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
        pl.read_parquet(utils.OUT_PATH / "posterior_ssp.parquet")
        .join(data, left_on="param", right_on="iso639p3code", how="inner")
        .join(langs, left_on="param", right_on="iso639p3code", how="inner")
        .with_columns(
            qtile=pl.col("median").qcut([0.5], labels=["Bottom half", "Top half"])
        )
        .sort(["qtile", "median"], descending=[False, True])
    )

    f = p.unique("family_id").get_column("family_id").sort().to_list()
    cm = plt.get_cmap("tab20")
    vals = cm(np.linspace(0, 1, len(f)), alpha=1)
    clrs = {k: colors.rgb2hex(v, keep_alpha=True) for k, v in zip(f, vals)}
    p = p.partition_by("qtile", as_dict=True)

    counter = defaultdict(int)

    fig, axes = plt.subplots(1, len(p), figsize=(12, 8), sharex=True)
    for i, d in enumerate(p.values()):
        ax = axes[i]

        for r in d.iter_rows(named=True):
            label = r["family_id"] if counter[r["family_id"]] == 0 else None
            c = clrs[r["family_id"]]
            y = languages.iso639_p3_to_name(r["param"])
            ax.hlines(
                y=y,
                xmin=r["lower"],
                xmax=r["upper"],
                colors=[c],
                linewidths=[1],
            )
            ax.scatter(r["median"], y, c=c, label=label, marker="|")
            ax.scatter(r["is_ssp"], y, c=c, marker="x", zorder=-1)
            counter[r["family_id"]] += 1

        ax.tick_params(axis="y", labelsize=6.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0%}"))
        ax.xaxis.grid(True, color="lightgrey", linestyle=":")

        ticks = ax.get_yticks()
        labels = [f"{p:.1%} ($N={n:5,d}$)" for p, n in zip(d["median"], d["n"])]
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(labels, ha="left")
        ax2.tick_params(axis="y", labelsize=5)

        # for row in d.iter_rows(named=True):
        #     label = row["family_id"] if counter[row["family_id"]] == 0 else None
        #     ax.scatter(0, row["param"], c=clrs[row["family_id"]], s=5, label=label)

        # counter[row["family_id"]] += 1
    fig.tight_layout()
    fig.legend(
        title=r"Language family",
        ncol=len(counter) // 3,
        alignment="center",
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.savefig(utils.IMG_PATH / "ssp-groups.png", dpi=DPI, bbox_inches="tight")


def plot_ssp_params(file: Path = Path("img/ssp-params.png")):
    params = ["theta", "gamma"]
    s = pl.read_parquet(utils.MODEL_PATH / "ssp" / "ssp.parquet").select(params)
    post = (
        s.median()
        .transpose(header_name="param", include_header=True)
        .rename({"column_0": "median"})
    )
    lower, upper = [], []
    for col in s.iter_columns():
        lo, up = model.hdi(col.to_numpy())
        lower.append(float(lo)), upper.append(float(up))
    post.insert_column(-1, pl.Series("lower", lower))
    post.insert_column(-1, pl.Series("upper", upper))

    fig, ax = plt.subplots(1, len(params), figsize=(8, 3))
    ax = ax.flatten()
    for idx, (r, p) in enumerate(zip(s.iter_columns(), post.iter_rows(named=True))):
        ax[idx].hist(r, color="orange", linewidth=3, edgecolor="white")

        _, ypos = ax[idx].get_ylim()
        ypos *= 1.1
        ax[idx].scatter(p["median"], ypos, s=50, c=COLOR)
        ax[idx].hlines(
            [ypos],
            [p["lower"]],
            [p["upper"]],
            linewidths=[5 / 2],
            colors=[COLOR],
        )
        label = f"{p['median']:.3f} [{p['lower']:.3f}, {p['median']:.3f}]"
        ax[idx].text(p["median"], ypos * 1.05, label, fontsize=9, ha="center")
        ax[idx].set_ylim(0, ypos * 1.2)
        ax[idx].set_title(f"$\\{p['param']}$")
        ax[idx].set_yticks([])

    fig.set_size_inches(8, 3)

    if file:
        fig.savefig(file, dpi=800)


def plot_freq_params(which: str = "subtlex", file: Path = Path("img/freq-params.png")):
    if which not in ["subtlex", "childes"]:
        raise ValueError("which must be one of 'subtlex' or 'childes'")

    params = ["beta_0", "beta_1", "pi_0", "pi_1"]
    s = pl.read_parquet(utils.OUT_PATH / f"syllables_freq_{which}.parquet")
    post = pl.read_parquet(
        utils.MODEL_PATH / f"freq_{which}" / f"freq_{which}.parquet"
    ).select(params)
    out = (
        post.median()
        .transpose(header_name="param", include_header=True)
        .rename({"column_0": "median"})
    )
    lower, upper, p_rope = [], [], []
    rope = np.std(list(s["freq"])) * -0.1, np.std(list(s["freq"])) * 0.1

    for col in post.iter_columns():
        if col.name == "param":
            continue
        lo, up = model.hdi(col.to_numpy())
        p = model.interval_overlap(rope, (lo, up))
        lower.append(float(lo))
        upper.append(float(up))
        p_rope.append(p)

    out.insert_column(-1, pl.Series("lower", lower))
    out.insert_column(-1, pl.Series("upper", upper))
    out.insert_column(-1, pl.Series("p_rope", p_rope))

    fig, ax = plt.subplots(1, len(params), figsize=(8, 3), tight_layout=True)
    for idx, (r, p) in enumerate(zip(post.iter_columns(), out.iter_rows(named=True))):
        ax[idx].hist(r, color="orange", lw=3, edgecolor="white")

        _, ypos = ax[idx].get_ylim()
        ypos *= 1.1
        ax[idx].scatter(p["median"], ypos, s=50, c=COLOR)
        ax[idx].hlines(
            [ypos],
            [p["lower"]],
            [p["upper"]],
            linewidths=[5 / 2],
            colors=[COLOR],
        )
        label = f"{p['median']:.2f} [{p['lower']:.2f}, {p['median']:.2f}]\n$p(ROPE)$ = {p['p_rope']:.2%}"
        # ax[idx].text(p["median"], ypos * 1.05, label, fontsize=9, ha="center")
        ax[idx].set_ylim(0, ypos * 1.2)
        ax[idx].set_yticks([])
        ax[idx].set_title(f"$\\{p['param']}$: {label}")

    fig.suptitle(f"Frequencies: {which.upper()}")
    fig.set_size_inches(8, 3)

    if file:
        fig.savefig(file, dpi=800)


def plot_freq(which: str = "subtlex"):
    if which not in ["subtlex", "childes"]:
        raise ValueError("which must be one of 'subtlex' or 'childes'")

    s = (
        pl.read_parquet(utils.OUT_PATH / f"syllables_freq_{which}.parquet")
        .group_by(["is_ssp", "iso639p3code"])
        .agg(freq=pl.col("freq").drop_nans().mean())
    )
    fig, ax = plt.subplots(1, 1)
    for r in s.iter_rows(named=True):
        ax.scatter(r["is_ssp"], r["freq"])

    for r in s.pivot(on="is_ssp", index="iso639p3code", values="freq").drop_nulls():
        pass


def main():
    plot_ssp_params()
    plot_ssp_langs()
    plot_freq_params(which="subtlex")
    plot_freq_params(which="childes")


if __name__ == "__main__":
    main()
