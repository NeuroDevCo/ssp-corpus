# ssp-corpus

[![Testing](https://github.com/NeuroDevCo/ssp-corpus/actions/workflows/lint-test.yml/badge.svg)](https://github.com/NeuroDevCo/ssp-corpus/actions/workflows/lint-test.yml)

Estimating the prevalence of the Sonority Sequencing Principle (SSP) across languages.

## Repository organisation

```
ssp-corpus/
├─ data/
│  ├─ raw/ # raw data, scraped from Wiktionary
│  ├─ processed/
│     ├─ words/ # processed word dataset for each language
│     ├─ words/ # processed syllables dataset for each language
├─ docs/ # materials needed to render the manuscript in Quarto
├─ img/ # figures for manuscript
├─ results/ # final datasets and model posterior
│  ├─ no-flat/ # results when classifying flat sonority contours as violating
├─ src/
│  ├─ languages.py # extract available languages from Glottolog
│  ├─ model.py # PyMC model
│  ├─ scrape.py # functions to scrape Wiktionary using wikipron
│  ├─ syllables.py # preprocess syllables
│  ├─ words.py # preprocess words
├─ tests/ # unit tests for syllable processing functions and utils
├─ _quarto.yml # Quarto global params
├─ main.py # main script to run everything in order
├─ README.md
````

## Pre-requisites

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Download file `data.zip` from the [OSF repository](https://osf.io/yeqms/files/osfstorage) and unzip into the project folder. Alternatively run the `src/scrap.py` script if you want to scrape Wiktionari anew using wikipron:

```bash
uv run src/scrape.py # add -l [code] to scrape a particular language
```

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


