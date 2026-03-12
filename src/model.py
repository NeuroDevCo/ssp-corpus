"""Model SSP prevalence."""

from collections.abc import Sequence
from json import dump
from pathlib import Path

import arviz as az
import arviz_stats as az_stats
from cmdstanpy import CmdStanModel, stanfit
import numpy as np
import polars as pl
import xarray as xr

RANDOM_SEED: int = 1234
rng = np.random.default_rng(RANDOM_SEED)


def prepare_data(
    syllables: pl.DataFrame,
    languages: pl.DataFrame,
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
    group_var: str | None = None,
    n_samples: int = 4_000,
    n_chains: int = 8,
    file: Path | str | None = None,
    **kwargs,
) -> tuple[stanfit.CmdStanMCMC, stanfit.CmdStanMCMC, dict]:
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
    y, n = [data[v].to_numpy() for v in (y_var, n_var)]
    stan_file = Path("model/model.stan")
    d = {"N": len(y), "n": n, "y": y}

    if group_var:
        lookup = get_group_indices(data[group_var].to_list())
        groups = data[group_var].unique().to_numpy()
        group_idx = [lookup[g] for g in data[group_var]]
        stan_file = Path("model/model_groups.stan")
        d.update({"J": len(groups), "j": group_idx, "group_labels": lookup})

    m = CmdStanModel(stan_file=stan_file)

    prior_fit = m.sample(
        data=d,
        fixed_param=True,  # sample from priors only
        iter_sampling=int(n_samples // 1.25),
        iter_warmup=int(n_samples // 4),
        seed=RANDOM_SEED,
        chains=n_chains,
        parallel_chains=n_chains,
    )

    fit = m.sample(
        data=d,
        step_size=0.1,
        iter_sampling=int(n_samples // 1.25),
        iter_warmup=int(n_samples // 4),
        seed=RANDOM_SEED,
        chains=n_chains,
        parallel_chains=n_chains,
        **kwargs,
    )

    if file:
        fit.save_csvfiles(str(file))

    return fit, prior_fit, d


def serialize(x):
    if x is None:
        return

    if isinstance(x, np.int64):
        return float(x)
    elif isinstance(x, xr.DataArray | xr.Dataset):
        if not len(x):
            return []
        return np.ndarray(x)
    elif isinstance(x, np.ndarray):
        return list(x)
    else:
        return str(x)


def diagnose_model(
    fit: stanfit.CmdStanMCMC,
    which: str | Sequence[str] | None = None,
    file: Path | str = Path("model/diagnostics.json"),
):
    idata = az.from_cmdstanpy(fit)
    is_ok, diag = az_stats.diagnose(
        idata, show_diagnostics=False, return_diagnostics=True
    )

    if isinstance(which, str):
        which = [which]

    if which:
        diag = {k: v for k, v in diag.items()}

    with Path(file).open(mode="w") as f:
        dump(diag, f, default=lambda o: serialize(o))

    return is_ok, diag


def summarize_posterior(
    fit: stanfit.CmdStanMCMC,
    stan_data: dict,
    param: str = "alpha",
    width: float = 0.95,
    file: Path | str = Path("out", "posterior-families.parquet"),
    **kwargs,
) -> pl.DataFrame:
    """Summarize posterior of the model.

    Args:
        fit (stanfit.CmdStanMCMC): Posterior samples of the mode.
        stan_data (dict): Stan data.
        width (float): Confidence level for the intervals.

    Returns:
        stanfit.CmdStanMCMC: Summarised posterior.
    """
    idata = az.from_cmdstanpy(fit)
    p = az_stats.summary(idata, ci_prob=width, ci_kind="hdi", **kwargs)
    p.index.set_names(["param"], inplace=True)
    p.reset_index(inplace=True)

    draws = pl.DataFrame(pl.from_pandas(p)).filter(
        pl.col("param").str.starts_with(param)
    )

    if "group_labels" in stan_data:
        lookup = {str(v - 1): k for k, v in stan_data["group_labels"].items()}
        draws = draws.with_columns(
            pl.col("param")
            .str.replace_all(f"\\[|\\]|{param}", "")
            .replace_strict(lookup, default=pl.col("param"))
        )

    float_cols = [c for c in draws.columns if c != "param"]
    int_cols = [c for c in draws.columns if c.startswith("ess")]
    draws = (
        draws.with_columns(pl.col(float_cols).cast(pl.Float64))
        .with_columns(pl.col(int_cols).cast(pl.Float64))
        .sort("mean", descending=True)
    )

    if file:
        file = Path(file)

    return draws


if __name__ == "__main__":
    from pathlib import Path

    import arviz as az
    import arviz_plots as azp
    import arviz_stats as azs
    from cmdstanpy import CmdStanModel
    from matplotlib import pyplot as plt
    import numpy as np
    import polars as pl
    import xarray as xr

    RANDOM_SEED: int = 1234
    rng = np.random.default_rng(RANDOM_SEED)

    N: int = 100

    iso639p3code = range(N)
    family_id = np.random.choice(range(N // 10))
    n = np.random.randint(100, 5_000, size=N)

    PROP = 0.5
    PROP_SD = 0.2
    prop = np.random.normal(PROP, PROP_SD, N)
    y = np.array([np.random.binomial(n, p, 1) for p in prop]).T
    data = {"iso639p3code": iso639p3code, "family_id": family_id, "n": n, "sum": y}

    df = pl.DataFrame(data)
    stan_data = {"N": len(y), "n": n, "y": y}

    stan_file = Path("model/model.stan")
    model = CmdStanModel(stan_file=stan_file)

    samples_prior = model.sample(
        data=data,
        fixed_param=True,
        iter_sampling=1_000,
        iter_warmup=1_000,
        seed=RANDOM_SEED,
        chains=8,
        parallel_chains=8,
    )

    samples = model.sample(
        data=data,
        iter_sampling=1_000,
        iter_warmup=1_000,
        seed=RANDOM_SEED,
        chains=8,
        parallel_chains=8,
    )

    idata = az.from_cmdstanpy(
        posterior=samples,
        prior=samples_prior,
        posterior_predictive="y_rep",
        prior_predictive="y_rep_prior",
        observed_data={"y": data["y"]},
    )

    az.plot_ppc(
        idata, data_pairs={"obs": "obs"}, alpha=0.03, figsize=(12, 6), textsize=14
    )

    posterior = az.extract(idata, group="posterior", var_names="theta", combined=True)
    posterior_hdi = azs.hdi(idata)
    posterior_mean = posterior.mean("sample")

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(idata.observed_data, density=True, stacked=True)
    ax[1].hist(prop, color="tab:red", density=True, stacked=True)

    # az.plot_trace(idata, var_names="alpha", compact=False)
    pc = azp.plot_ppc_dist(idata, group="y_rep", kind="hist")
