"""Model SSP data."""

import numpy as np
import polars as pl
import pymc as pm

RANDOM_SEED = 1234
rng = np.random.default_rng(RANDOM_SEED)


def prepare_data(
    data: pl.DataFrame, y_var: str = "is_ssp", group_var: str = "lang_family"
) -> tuple[np.ndarray]:
    """Prepare data for modelling.

    Args:
        data (pl.DataFrame): Dataset with response variable and predictors.
        y_var (str, optional): Name of the response variable in the DataFrame. Defaults to "prop_sspp".
        group_var (str, optional): Name of the grouping variable in the DataFrame. Degaults to "lang".

    Returns:
        tuple[np.ndarray]: Response variable, number of observations per group, and group index.
    """
    data = (
        data.select(pl.col([group_var, y_var]))
        .group_by(group_var)
        .agg(pl.len().alias("n"), pl.sum(y_var))
    )
    y = data.get_column(y_var).to_numpy()
    n = data.get_column("n").to_numpy()
    groups = data.get_column(group_var).to_numpy()
    return y, n, groups


def fit_model(y: np.ndarray, n: np.ndarray, groups: np.ndarray) -> tuple:
    """Fit multilevel Bayesian model.

    Args:
        y (np.ndarray): Response variable.
        n (np.ndarray): Number of observations per group.
        groups (np.ndarray): Group index.

    Returns:
        tuple: Model, inference data, and likelihood.
    """
    with pm.Model(coords={"group": groups}) as m:
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Exponential("sigma_a", 4)
        alpha = pm.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="group")
        p = pm.math.invlogit(alpha)
        y_like = pm.Binomial("y_like", n=n, p=p, observed=y)
        trace = pm.sample(2000, tune=2000)
    return m, trace, y_like


def invlogit(x: float) -> float:
    """Calculate inverse logit function"""
    odds = np.exp(x)
    return odds / (1 + odds)
