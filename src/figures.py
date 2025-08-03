"""Create figures."""

from pathlib import Path
import polars as pl
from matplotlib import pyplot as plt
import plotly.express as px
from src import pron


def plot_structures(
    datapath: str | Path = Path("out", "syllables.csv"),
    file: str | Path = Path("img", "map.png"),
):
    df = pl.read_csv(datapath).join(
        pron.get_languages().rename({"iso639p3code": "lang"}), on="lang"
    )
    fn = "family_name"
    fig, axes = plt.subplots(nrows=len(df.unique(fn)))
    for fam, ax in zip(df.get_column(fn).unique().to_numpy(), axes):
        d = df.filter(pl.col(fn) == fam).sort("n", descending=True)
        ax.scatter(d.get_column("struct"), d.get_column("n"), s=1)
    return None


def plot_n_words(
    datapath: str | Path = Path("out", "words.csv"),
    file: str | Path = Path("img", "map.png"),
):
    """Plot the distribution of distinct word-forms extracted for each language."""


def plot_lang_map(file: str | Path = Path("img", "map.png"), show: bool = True):
    """Make map of scraped languages.
    Args:
        file (str | Path, optional): Path to the generated PNG. Defaults to `Path("img", "map.png")`.
        show (bool, optional): Should the plot be shown? Defaults to True.
    """
    df = (
        pl.read_csv(Path("out", "words.csv"))
        .group_by(pl.col("lang"))
        .agg(pl.len().alias("n_pron"))
        .join(pron.get_languages().rename({"iso639p3code": "lang"}), on="lang")
        .filter(pl.col("pop_size") > 0)
        .with_columns(
            pl.col("pop_size").sqrt().fill_nan(None).alias("exp_pop_size"),
            pl.col("n_pron").sqrt().fill_nan(None).alias("exp_n_pron"),
        )
        .drop_nulls(pl.col("exp_pop_size"))
    )

    fig1 = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        size="exp_pop_size",
        color="family_name",
        projection="kavrayskiy7",
    )
    fig1.update_traces(
        marker={
            "color": "rgba(255, 0, 0, 0)",
            "line": {"width": 1, "color": "#333232"},
        },
        showlegend=False,
    ).select_traces()
    fig2 = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        size="exp_n_pron",
        color="family_name",
        opacity=0.5,
        hover_name="language_name",
        projection="kavrayskiy7",
    )
    fig2.update_traces(marker={"line": {"width": 0}})
    fig2.update_legends()
    fig3 = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color_discrete_sequence=["#333232"],
        hover_name="language_name",
        projection="kavrayskiy7",
    )
    fig3.update_traces(marker=dict(size=2.5))
    fig = px.scatter_geo()
    fig.add_traces(fig1._data)
    fig.add_traces(fig2._data)
    fig.add_traces(fig3._data)
    fig.update_layout(map_style="carto-positron-nolabels")

    if show:
        fig.show()
    fig.write_image(file)


if __name__ == "__main__":
    plot_lang_map()
