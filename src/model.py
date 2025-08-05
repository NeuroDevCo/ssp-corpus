"""Model SSP data."""

from pathlib import Path
from functools import lru_cache
import numpy as np
import polars as pl
import pymc as pm
from pytensor import tensor as pt
import arviz as az
from scipy.stats import gaussian_kde
from src import scraping

RANDOM_SEED = 1234
rng = np.random.default_rng(RANDOM_SEED)


def hdi(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode

    Note:
        Taken from https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py

    Args:
        sample (np.ndarray | list): An array containing MCMC samples.
        alpha (float, optional): Desired probability of type I error. Defaults to 0.05.
        roundto (int): Number of digits after the decimal point for the results. Defaults to 2.

    Returns:
        hpd: array with the lower
    """  # pylint: disable=line-too-long
    sample = sample[~np.isnan(np.asarray(sample))]
    # get upper and lower bounds
    lo, up = np.min(sample), np.max(sample)
    density = gaussian_kde(sample)
    x = np.linspace(lo, up, 2000)
    y = density.evaluate(x)
    xy_zipped = zip(x, y / np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1 - alpha):
            break
    hdv.sort()
    diff = (up - lo) / 20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i - 1] >= diff:
            hpd.append(round(hdv[i - 1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    return hpd


def proportion_by(data: pl.DataFrame, by: str = None) -> pl.DataFrame:
    """Get proportion by group."""
    if by is None:
        by = ["lang", "family_name", "language_name"]
    langs = (
        scraping.get_languages()
        .rename({"iso639p3code": "lang"})
        .select(pl.col(["lang", "family_name", "language_name"]))
    )
    return (
        data.join(langs, on="lang")
        .drop_nulls(["family_name"])
        .group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.mean("is_ssp").alias("prop"),
            pl.sum("is_ssp").alias("sum"),
        )
    )


def fit_model(
    data: pl.DataFrame,
    y_var: str = "sum",
    pred_var: str = None,
    group_var: str = "family_name",
    nsamples: int = 4_000,
    file: str | Path = None,
) -> tuple:
    """Fit multilevel Bayesian model.

    Args:
        data (pl.DataFrame): Dataset with response variable and predictors.
        y_var (str, optional): Name of the response variable in the DataFrame. Defaults to "prop_sspp".
        group_var (str, optional): Name of the grouping variable in the DataFrame. Degaults to "lang".

    Returns:
        tuple: Model, inference data, and likelihood.
    """
    if file is None:
        fn = f"model_{pred_var}.nc" if pred_var else "model_base.nc"
        file = Path("out", fn)
    y = data.get_column(y_var).to_numpy()
    n = data.get_column("n").to_numpy()
    groups = data.get_column(group_var).to_numpy()
    groups_fact = np.unique(np.arange(len(groups)), return_inverse=True)[1]
    if pred_var:
        x = data.get_column(pred_var).to_numpy()
        x_std = (x - x.mean()) / x.std()
    with pm.Model(coords={"group": groups}) as m:
        group_idx = pm.Data("group_idx", groups_fact, dims="obs_id")
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Exponential("sigma_a", 4)
        theta = pm.Normal("theta", mu=mu_a, sigma=sigma_a, dims="group")
        if pred_var:
            beta = pm.Normal("beta", mu=0.0, sigma=10.0)
            p = pm.Deterministic("p", pm.math.invlogit(theta[group_idx] + beta * x_std))
        else:
            p = pm.Deterministic("p", pm.math.invlogit(theta[group_idx]))
        y_pred = pm.Binomial("y_pred", n=n, p=p, observed=y, dims="obs_id")
        prior = pm.sample_prior_predictive(samples=nsamples)
        kw = {"log_likelihood": True}
        trace = pm.sample(nsamples // 2, tune=nsamples // 2, idata_kwargs=kw)
    return data, m, trace, prior, y_pred


def invlogit(x: float) -> float:
    """Calculate inverse logit function"""
    odds = np.exp(x)
    return odds / (1 + odds)


@lru_cache
def summarize_posterior(idata: az.InferenceData) -> pl.DataFrame:
    d = az.convert_to_dataset(idata).to_dataframe().reset_index()
    p = (
        pl.from_pandas(d)
        .drop_nulls()
        .rename({"group": "family_name"})
        .with_columns(
            pl.col("alpha").map_elements(invlogit, return_dtype=pl.Float64),
        )
        .group_by("family_name")
        .agg(
            pl.col("alpha").mean().alias("mean"),
            pl.struct("alpha")
            .map_batches(
                lambda x: hdi(x.to_numpy())[0], return_dtype=pl.List(pl.Float64)
            )
            .first()
            .alias("hdi"),
        )
        .sort("mean", descending=True)
        .with_columns(
            pl.col("hdi").list.first().alias("lower"),
            pl.col("hdi").list.last().alias("upper"),
        )
        .select(["family_name", "mean", "lower", "upper"])
    )
    return p


def compare_models(x: dict[str, az.InferenceData]):
    models = {k: v[2] for k, v in x.items()}
    return az.compare(models, method="BB-pseudo-BMA")
