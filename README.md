# ssp-corpus

[![Testing](https://github.com/NeuroDevCo/ssp-corpus/actions/workflows/lint-test.yml/badge.svg)](https://github.com/NeuroDevCo/ssp-corpus/actions/workflows/lint-test.yml)

Estimating the prevalence of the Sonority Sequencing Principle (SSP) across languages.

## Pre-requisites

This project uses 

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Download file `data.zip` from the [OSF repository](https://osf.io/yeqms/files/osfstorage) and unzip into the project folder.


## Running the code

```bash
uv sync
uv run main.py -l -w -s -m -v -d
```

Where the flags refer to the following actions (each action depends on the previous one):

* `-l` or `--languages`: generate language dataframe
* `-w` or `--words`: generate words dataframe
* `-s` or `--syllables`: generate words dataframe
* `-m` or `--model`: run Bayesian models in Stan
* `-v` or `--viz`: generate plots
* `-d` or `--docs`: generate manuscript in Quarto


