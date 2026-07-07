"""Generate figures."""

from collections import defaultdict
from pathlib import Path
import polars as pl

from great_tables import GT
from matplotlib import pyplot as plt
from matplotlib import colors, ticker
import numpy as np

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
        pl.read_parquet(utils.RESULTS_PATH / "languages.parquet")
        .unique("family_id")
        .select("family_id")
        .sort("family_id")
    )
    f = langs.get_column("family_id").sort().to_list()
    cm = plt.get_cmap(cmap)
    vals = cm(np.linspace(0, 1, len(f)), alpha=alpha)

    return {k: colors.rgb2hex(v, keep_alpha=True) for k, v in zip(f, vals)}


def plot_ssp_langs():
    s = pl.read_parquet(utils.RESULTS_PATH / "syllables.parquet")
    langs = (
        pl.read_parquet(utils.RESULTS_PATH / "languages.parquet")
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
        pl.read_parquet(utils.RESULTS_PATH / "model-posterior" / "ssp.parquet")
        .with_columns(pl.col("param").str.replace_many(["theta_j[", "]"], ["", ""]))
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

    fig, axes = plt.subplots(1, len(p), sharex=True)
    for i, d in enumerate(p.values()):
        ax = axes[i]

        for r in d.iter_rows(named=True):
            y = languages.iso639_p3_to_name(r["param"])
            ax.hlines(
                y=y,
                xmin=r["lower"],
                xmax=r["upper"],
                colors=[COLOR],
                linewidths=[1],
            )
            ax.scatter(r["median"], y, c=COLOR, marker="|")
            ax.scatter(r["is_ssp"], y, c="orange", marker="x", zorder=-1)

            label = r["family_id"] if counter[r["family_id"]] == 0 else None
            ax.scatter(0, y, c=clrs[r["family_id"]], s=10, label=label)
            counter[r["family_id"]] += 1

            ax.text(0.025, y, c="k", fontsize=7, s=y, va="center")

        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.xaxis.grid(True, color="lightgrey", linestyle=":")

        ticks = ax.get_yticks()
        labels = [f"{p:.1%} ($N={n:5,d}$)" for p, n in zip(d["median"], d["n"])]
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(labels, ha="left")
        ax2.tick_params(axis="y", labelsize=5)

    fig.supxlabel("Proportion of SSP-following syllables (%)", y=0.15)
    fig.tight_layout()
    fig.legend(
        title=r"Language family",
        ncol=len(counter) // 3,
        alignment="center",
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.125),
    )
    fig.set_size_inches(8, 8)
    fig.savefig(utils.IMG_PATH / "ssp-groups.png", dpi=DPI, bbox_inches="tight")


def plot_freq_distribution():
    sfs = pl.read_parquet(
        utils.RESULTS_PATH / "syllables_freq_subtlex.parquet"
    ).with_columns(source=pl.lit("Adults"))
    sfc = pl.read_parquet(
        utils.RESULTS_PATH / "syllables_freq_childes.parquet"
    ).with_columns(source=pl.lit("CHILDES"))
    sf = pl.concat([sfs, sfc])

    fig, axes = plt.subplots(2, 2, sharex=True)
    for widx, which in enumerate(set(sf["source"])):
        dl = sf.filter(pl.col("source") == which)
        for idx, d in enumerate(dl.partition_by("is_ssp")):
            ax = axes[widx, idx]
            ax.hist(d["freq"], lw=2, edgecolor="white", bins=20, color="orange")
            n = len(d)
            title = (
                f"{which}: SSP-following"
                if all(ssp for ssp in d["is_ssp"])
                else f"{which}: SSP-violating"
            )
            title += f"\n($n$ = {n:,.0f})"
            ax.set_title(title)
            ax.xaxis.grid(True, color="lightgrey", linestyle=":")

            ticks = ax.get_yticks()
            ax.set_yticks(ticks)
            labels = [f"{t:,.0f}" for t in ticks]
            ax.set_yticklabels(labels, ha="right", fontsize=5)

    fig.supxlabel("Syllabic frequency (Zipf)")
    fig.supylabel("$N$ of syllables")
    fig.tight_layout()
    fig.set_size_inches(6, 5)
    fig.savefig(utils.IMG_PATH / "freq-distribution.png", dpi=800)


def plot_ssp_bias():
    s = pl.read_parquet(utils.RESULTS_PATH / "syllables.parquet")

    data = (
        s.group_by(["iso639p3code"])
        .agg(pl.col("is_ssp").mean(), n=pl.len().log())
        .with_columns(
            se=((pl.col("is_ssp") * (1 - pl.col("is_ssp"))) / pl.col("n")).sqrt()
        )
    )

    fig, ax = plt.subplots(1, 1)
    ax.scatter(data["n"], data["is_ssp"], color="orange", s=10)
    dist_to_limit = data["is_ssp"] - 1
    yerr_clipped = np.maximum(data["se"], dist_to_limit)
    ax.errorbar(
        data["n"],
        data["is_ssp"],
        yerr=yerr_clipped,
        c="orange",
        alpha=1 / 4,
        linestyle="None",
    )

    coef = np.polyfit(data["n"].to_list(), data["is_ssp"].to_list(), 1)
    fit = np.poly1d(coef)

    ax.plot(data["n"], fit(data["n"]), lw=3, c=COLOR)
    ax.set_xlabel("Log(Number of syllables)")
    ax.set_ylabel("Proportion of SSP-following syllables")

    _, ymax = ax.get_ylim()
    ax.scatter(
        data["n"],
        [ymax + (ymax * 0.025)] * len(data),
        marker="|",
        c="orange",
        alpha=0.5,
    )

    _, xmax = ax.get_xlim()
    ax.scatter(
        [xmax + (xmax * 0.025)] * len(data),
        data["is_ssp"],
        marker="_",
        c="orange",
        alpha=0.5,
    )

    ticks = np.linspace(0.5, 1, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.0%}"))

    fig.savefig(Path("img/ssp-bias.png"), dpi=800)


def plot_ssp_params(file: Path = Path("img/ssp-params.png")):
    params = ["theta", "gamma"]
    data_path = utils.RESULTS_PATH / "model-posterior" / "samples_ssp.parquet"
    s = pl.read_parquet(data_path).select(params)
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
        xpos = ax[idx].get_xlim()
        xpos = sum(xpos) / len(xpos)
        ax[idx].scatter(p["median"], ypos, s=50, c=COLOR)
        ax[idx].hlines(
            [ypos],
            [p["lower"]],
            [p["upper"]],
            linewidths=[5 / 2],
            colors=[COLOR],
        )
        label = f"{p['median']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]"
        ax[idx].set_ylim(0, ypos * 1.2)
        ax[idx].set_yticks([])

        ax[idx].set_title(f"$\\{p['param']}$\n", weight="bold")
        ax[idx].text(xpos, ypos * 1.25, label, ha="center", fontsize=8)

    fig.set_size_inches(8, 3)

    if file:
        fig.savefig(file, dpi=800)


def plot_freq_params(which: str = "subtlex"):
    if which not in ["subtlex", "childes"]:
        raise ValueError("which must be one of 'subtlex' or 'childes'")

    params = ["beta_1"]
    post = pl.read_parquet(
        utils.RESULTS_PATH / "model-posterior" / f"samples_freq_{which}.parquet"
    ).select(params)
    out = (
        post.median()
        .transpose(header_name="param", include_header=True)
        .rename({"column_0": "median"})
    )
    lower, upper = [], []

    for col in post.iter_columns():
        if col.name == "param":
            continue
        lo, up = model.hdi(col.to_numpy())
        lower.append(float(lo))
        upper.append(float(up))

    out.insert_column(-1, pl.Series("lower", lower))
    out.insert_column(-1, pl.Series("upper", upper))

    fig, ax = plt.subplots(1, len(params), figsize=(8, 3), tight_layout=True)
    for r, p in zip(post.iter_columns(), out.iter_rows(named=True)):
        ax.hist(r, color="orange", lw=3, edgecolor="white")

        _, ypos = ax.get_ylim()
        ypos *= 1.1
        xpos = ax.get_xlim()
        xpos = sum(xpos) / len(xpos)

        ax.scatter(p["median"], ypos, s=50, c=COLOR)
        ax.hlines(
            [ypos],
            [p["lower"]],
            [p["upper"]],
            linewidths=[5 / 2],
            colors=[COLOR],
        )
        ax.axvline(0, linestyle="dotted", color="grey")
        label = f"{p['median']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]"
        ax.set_ylim(0, ypos * 1.2)
        ax.set_yticks([])
        ax.set_title(f"SSP-following $-$ SSP-violating ($\\{p['param']}$)\n")
        ax.text(xpos, ypos * 1.25, label, ha="center", fontsize=8)

    fig.suptitle(f"Frequencies: {which.upper()}")
    fig.set_size_inches(4, 4)

    fn = Path("img") / f"freq-params-{which}.png"
    fig.savefig(fn, dpi=800)


def plot_freq(which: str = "subtlex"):
    if which not in ["subtlex", "childes"]:
        raise ValueError("which must be one of 'subtlex' or 'childes'")

    s = (
        pl.read_parquet(utils.RESULTS_PATH / f"syllables_freq_{which}.parquet")
        .group_by(["is_ssp", "iso639p3code"])
        .agg(freq=pl.col("freq").drop_nans().mean())
    )
    fig, ax = plt.subplots(1, 1)
    for r in s.iter_rows(named=True):
        ax.scatter(r["is_ssp"], r["freq"])

    for r in s.pivot(on="is_ssp", index="iso639p3code", values="freq").drop_nulls():
        pass


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
    (
        GT(df, groupname_col="is_ssp")
        .cols_label(
            lang="Language",
            syll="Syllable",
            phon="Original word",
            sonority="Sonority index",
            sonority_2="",
        )
        .fmt_nanoplot(columns="sonority_2")
    ).gtsave(Path("img/examples.png"), zoom=10)


def main():
    table_examples()
    plot_ssp_params()
    plot_ssp_langs()
    plot_freq_distribution()
    plot_freq_params(which="subtlex")
    plot_freq_params(which="childes")


if __name__ == "__main__":
    main()
