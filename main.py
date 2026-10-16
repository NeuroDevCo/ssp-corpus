"""Main script."""

import argparse
from pathlib import Path

import quarto
from rich.console import Console

from src import languages, model, syllables, utils, viz, words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ssp-corpus")

    parser.add_argument("-l", "--langs", action="store_true")
    parser.add_argument("-w", "--words", action="store_true")
    parser.add_argument("-s", "--syllables", action="store_true")
    parser.add_argument("-m", "--model", action="store_true")
    parser.add_argument("-v", "--viz", action="store_true")
    parser.add_argument("-d", "--docs", action="store_true")

    args = parser.parse_args()

    con = Console()

    if args.langs:
        con.rule("Preparing Glottolog language dataset")
        langs = languages.main()

    if args.words:
        con.rule("Preprocessing word forms")
        w = words.main()

    if args.syllables:
        con.rule("Extracting syllables")
        s = syllables.main()

    if args.model:
        con.rule("Fitting Bayesian models")
        model.main(
            utils.RESULTS_PATH,
            target_accept=0.95,
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
        )
        model.main(
            utils.RESULTS_PATH / "no-flat",
            target_accept=0.95,
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
        )

    if args.viz:
        con.rule("Making figures")
        viz.main()
        p = utils.RESULTS_PATH / "no-flat"
        viz.main(
            data_path=p / "syllables.parquet",
            post_path=p / "post.nc",
            img_path=utils.IMG_PATH / "no-flat",
        )

    if args.docs:
        con.rule("Rendering documents")
        quarto.render(Path())
