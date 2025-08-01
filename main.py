#!/usr/bin/env python3

"""
Main script of the project.
"""

from pathlib import Path
import polars as pl
from src import corpus, model

if __name__ == "__main__":
    WORDS, SYLL = corpus.get_corpora()
    SSP = (
        SYLL.group_by("lang").agg(pl.mean("is_ssp").alias("prop_ssp")).sort("prop_ssp")
    )
    SSP.write_csv(Path("out", "ssp.csv"))

    DATA = model.prepare_data(SSP)
    MOD, TRACE, Y_LIKE = model.fit_model(*DATA)
    model.plot_estimations(idata=TRACE)
