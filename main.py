#!/usr/bin/env python3

"""
Main script of the project.
"""

from pathlib import Path
import polars as pl
from matplotlib import pyplot as plt


if __name__ == "__main__":
    data = []
    for f in Path("data", "processed").glob("*.csv"):
        lang = f.name.rsplit(".", 1)[0]
        df = pl.read_csv(f)
        df = df.with_columns(pl.lit(lang).alias("lang"))
        data.append(df)
    data = pl.concat(data)

    agg = (
        data.group_by(pl.col(["lang", "struct", "is_ssp"]))
        .len()
        .with_columns((pl.col("len") / pl.sum("len")).alias("percent"))
    )
