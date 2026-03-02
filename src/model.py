"""Model SSP prevalence."""

from collections.abc import Sequence
from pathlib import Path

import arviz as az
from cmdstanpy import CmdStanModel, stanfit
import numpy as np
import polars as pl

RANDOM_SEED: int = 1234
rng = np.random.default_rng(RANDOM_SEED)


def get_group_indices(x: Sequence[str]) -> dict[str, int]:
    """Assign integer indicaes to each group.

    Args:
        data (list[str]): List of values to recode.

    Returns:
        dict[str, int]: Recoded values.
    """
    vals = sorted(list(set(x)))
    return dict(zip(vals, range(1, len(vals) + 1)))


def fit_model(
    data: pl.DataFrame,
    y_var: str = "sum",
    n_var: str = "n",
    group_var: str = "family_name",
    n_samples: int = 4_000,
    n_chains: int = 8,
    file: Path | str = Path("model", "model.stan"),
    outpath: Path | str | None = None,
    **kwargs,
) -> tuple[stanfit.CmdStanMCMC, dict]:
    """Fit Bayesian multilevel model with a Binomial distribution in CmdStanPy.

    Args:
        data (pl.DataFrame): Data to model.
        y_var (str, optional): Name of the response variable in ``data``. Defaults to "sum".
        n_var (str, optional): Name of the variable with number of trials in ``data``. Defaults to "n".capitalize
        group_var (str, optional): Name of the grouping variable in ``data``. Defaults to "lang".
        n_samples (int, optional): Total number of samples per MCMC chain. Defaults to 4_000.
        n_chains (int, optional): Number of MCMC chains to run. Defaults to 8.
        file (str | Path): Path to the Stan file with the model. Defaults to ``Path("model", "model.stan")``.
        outpath (str, optional): Path to the directory where the posterior samples of the model will be saved. Defaults to None.
        **kwargs: Additional arguments passed to the sampler.

    Returns:
        stanfit.CmdStanMCMC: Posterior MCMC samples.
        dict: Stan data.
    """
    file = Path(file)

    d = data.sort("family_name")
    y, n = [d[v].to_numpy() for v in (y_var, n_var)]
    lookup = get_group_indices(d[group_var].to_list())
    groups = d[group_var].unique().to_numpy()
    group_idx = [lookup[g] for g in d[group_var]]

    d = {
        "N": len(y),
        "n": n,
        "y": y,
        "J": len(groups),
        "j": group_idx,
        "group_labels": lookup,
    }

    m = CmdStanModel(stan_file=file)

    fit = m.sample(
        data=d,
        step_size=0.1,
        iter_sampling=n_samples // 2,
        iter_warmup=n_samples // 2,
        seed=RANDOM_SEED,
        chains=n_chains,
        parallel_chains=n_chains,
        **kwargs,
    )

    if outpath:
        fit.save_csvfiles(str(outpath))
    return fit, d


def summarize_posterior(
    fit: stanfit.CmdStanMCMC, stan_data: dict, width: float = 0.95
) -> pl.DataFrame:
    """Summarize posterior of the model.

    Args:
        fit (stanfit.CmdStanMCMC): Posterior samples of the mode.
        stan_data (dict): Stan data.
        width (float): Confidence level for the intervals.

    Returns:
        stanfit.CmdStanMCMC: Summarised posterior.
    """
    hdi_names = np.array(((1 - width) / 2, 1 - (1 - width) / 2)) * 100
    hdi_names = np.round(hdi_names, decimals=2)
    hdi_names = {f"hdi_{h}%": i for h, i in zip(hdi_names, ("lower", "upper"))}
    idata = az.from_cmdstanpy(fit)
    p = az.summary(idata, hdi_prob=width, var_names=["theta"])
    p.reset_index("param")

    lookup = {str(v - 1): k for k, v in stan_data["group_labels"].items()}
    draws = (
        pl.DataFrame(pl.from_pandas(p))
        .rename({"hdi_2.5%": "lower", "hdi_97.5%": "upper"})
        .with_columns(
            pl.col("param")
            .str.replace_all("\\[|\\]|theta", "")
            .replace_strict(lookup, default=None),
            pl.col(["mean", "sd", "lower", "upper"]).map_elements(
                lambda x: 1 / (1 + np.exp(-x)), return_dtype=pl.Float64
            ),
        )
        .drop_nulls()
        .sort("mean", descending=True)
    )
    return draws
