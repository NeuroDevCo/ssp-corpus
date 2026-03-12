"""Main script."""

from arviz import from_cmdstanpy, plot_dist

from pathlib import Path

import arviz as az
import arviz_stats as az_stats

from src.languages import get_languages
from src.words import get_words
from src.syllables import get_syllables
from src.model import fit_model, prepare_data, summarize_posterior, diagnose_model

IMG_PATH: Path = Path("img")


def make_data():
    langs = get_languages()
    words = get_words(langs)
    sylls = get_syllables(words)

    return langs, words, sylls


if __name__ == "__main__":
    langs, words, sylls = make_data()

    # fit model with no grouping variables
    data = prepare_data(sylls, langs)
    posterior, prior, samples_data = fit_model(data)
    model_diag = diagnose_model(posterior, file=Path("model/diagnostics.json"))

    dt = az.from_cmdstanpy(
        posterior=posterior,
        prior=prior,
        prior_predictive={"y": "y_prior_pred"},
        posterior_predictive={"y": "y_rep"},
        log_likelihood={"y": "log_lik"},
        observed_data={"y": samples_data["y"]},
    )
    plt.plot(samples_data["y"])
    az.plot_ppc_dist(dt, num_samples=300)
