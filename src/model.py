from dataclasses import dataclass, asdict
from pathlib import Path

import arviz as az
import arviz_plots as azp
from cmdstanpy import CmdStanModel
import polars as pl

from src.languages import get_languages
from src.words import get_words
from src.syllables import get_syllables


@dataclass
class Conf:
    seed: int = 1234
    iter_sampling: int = 1_000
    iter_warmup: int = 1_000
    chains: int = 8
    parallel_chains: int = 8
    adapt_delta: float = 0.9
    step_size: float = 0.01


def get_data(
    syllables: pl.DataFrame | None = None,
    languages: pl.DataFrame | None = None,
    by: str | list[str] = "iso639p3code",
    file: Path | str = Path("out/data.parquet"),
) -> pl.DataFrame:
    """Aggregate data to get the proportion of SSP vollowing syllables by
    group.

    Args:
        syllables (pl.DataFrame): Syllable dataframe to aggregate.
        languages (pl.DataFrame): Language dataframe.
        by (str | Sequence, optional): Variables to group the aggregates by. Defaults to None.
        file (str | Path, optional): Path to the generated parquet file. Defaults to `Path("out/data.parquet")`.

    Returns:
        pl.DataFrame: Aggregated data, ready for modelling.
    """
    if isinstance(by, str):
        by = [by]

    if languages is None:
        languages = get_languages()
    if syllables is None:
        syllables = get_syllables()

    languages = languages.select(by)
    join_cols = set(languages.columns).intersection(set(syllables.columns))

    df = (
        syllables.join(languages, on=list(join_cols), how="inner")
        .drop_nulls(by)
        .group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.mean("is_ssp").alias("prop"),
            pl.sum("is_ssp").alias("sum"),
        )
    )

    file = Path(file)
    if file:
        df.write_parquet(file)

    return df


def get_stan_data(data: pl.DataFrame):

    stan_data = {
        "N": len(data),
        "n": data["n"].to_list(),
        "y": data["sum"].to_list(),
        "prop": data["prop"].to_list(),
    }

    if "family_id" in data.columns:
        families = {k: i + 1 for i, k in enumerate(set(data["family_id"]))}
        n_fam = len(set(data["family_id"]))
        new_vars = {"J": n_fam, "g": [families[f] for f in data["family_id"]]}
        stan_data.update(new_vars)

    return stan_data


def fit_model(
    data: dict, stan_file: Path | str = Path("model/model.stan"), conf: Conf = Conf()
):
    model = CmdStanModel(stan_file=str(stan_file))
    samples = model.sample(data=data, **asdict(conf))
    idata = az.from_cmdstanpy(
        posterior=samples,
        posterior_predictive="theta_rep",
        observed_data={"y": data["prop"]},
        log_likelihood={"y": "log_lik"},
    )
    return model, samples, idata


if __name__ == "__main__":
    langs = get_languages()
    words = get_words(langs)
    sylls = get_syllables(words)
    sylls_freq = get_syllables(words, add_frequencies=True)

    # complete pooling model
    data = get_data(sylls, langs)
    stan_data = get_stan_data(data)
    stan_file = Path("model/model.stan")
    model, samples, idata = fit_model(stan_data, stan_file)

    azp.plot_rank_dist(idata, var_names=["theta", "phi"])
    az.plot_ppc(idata, data_pairs={"y": "theta_rep"}, num_pp_samples=25)

    # now with partial pooling by family
    data = get_data(sylls, langs, by=["iso639p3code", "family_id"])
    stan_data = get_stan_data(data)
    stan_file = Path("model/model-family.stan")
    model_family, samples_family, idata_family = fit_model(stan_data, stan_file)

    azp.plot_rank_dist(idata_family, var_names=["theta", "phi"])
    az.plot_ppc(idata_family, data_pairs={"y": "theta_rep"}, num_pp_samples=25)

    cmp_df = az.compare(
        {"model": idata, "model_family": idata_family}, method="BB-pseudo-BMA"
    )
    az.plot_compare(cmp_df)

    # complete pooling, only frequencies
    data = get_data(sylls_freq, langs, by=["iso639p3code"])
    stan_data = get_stan_data(data)
    stan_file = Path("model/model.stan")
    model_freq, samples_freq, idata_freq = fit_model(stan_data, stan_file)

    azp.plot_rank_dist(idata_freq, var_names=["theta", "phi"])
    az.plot_ppc(idata_freq, data_pairs={"y": "theta_rep"}, num_pp_samples=25)

    # partial pooling, only frequencies
    data = get_data(sylls_freq, langs, by=["iso639p3code", "family_id"])
    stan_data = get_stan_data(data)
    stan_file = Path("model/model.stan")
    model_freq_family, samples_freq_family, idata_freq_family = fit_model(
        stan_data, stan_file
    )

    azp.plot_rank_dist(idata_freq_family, var_names=["theta", "phi"])
    az.plot_ppc(idata_freq, data_pairs={"y": "theta_rep"}, num_pp_samples=25)

    cmp_df = az.compare(
        {"model_freq": idata_freq, "model_freq_family": idata_freq_family},
        method="BB-pseudo-BMA",
    )
    az.plot_compare(cmp_df)
