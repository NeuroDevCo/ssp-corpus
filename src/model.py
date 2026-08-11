import arviz_stats as az_stats
import numpy as np
import polars as pl
import pymc as pm
import xarray as xr

from src import utils


def get_data(data: pl.DataFrame) -> dict:
    j = data["iso639p3code"].cast(pl.Categorical)

    return {
        "y": data["is_ssp"].to_numpy().astype(np.int8),
        "x": data["freq"].to_numpy(),
        "j": j.to_physical().to_numpy().astype(int),
        "j_names": j.unique().sort().to_numpy(),
        "N": len(data),
        "J": j.n_unique(),
    }


def fit_model_nofreq(x: pl.DataFrame, **kwargs):
    data = get_data(x)
    coords = {"j": data["j_names"], "i": np.arange(data["N"])}
    y, j = data["y"], data["j"]  # No need to standardize x (unused)

    with pm.Model(coords=coords) as model:
        # Global intercept
        α = pm.Normal("α", mu=0, sigma=1)

        # Group-level random intercepts (non-centered)
        u_σ = pm.HalfNormal("u_σ", sigma=0.5)
        u_z = pm.Normal("u_z", mu=0, sigma=1, dims="j")
        u = pm.Deterministic("u", u_z * u_σ, dims="j")

        # Linear predictor (per observation)
        μ = pm.Deterministic("μ", α + u[j], dims="i")

        # Likelihood
        pm.Bernoulli("y", p=pm.math.sigmoid(μ), observed=y, dims="i")

        # Sample
        idata = pm.sample(**kwargs)
        pm.compute_log_likelihood(idata)

    return model, idata


def fit_model_freq(x: pl.DataFrame, **kwargs):
    data = get_data(x)
    coords = {"j": data["j_names"], "i": np.arange(data["N"])}
    x = (data["x"] - data["x"].mean()) / data["x"].std()  # Standardize
    y, j = data["y"], data["j"]

    with pm.Model(coords=coords) as model:
        # Global intercept and slope
        α = pm.Normal("α", mu=0, sigma=1)
        β = pm.Normal("β", mu=0, sigma=1)

        # Random intercepts (non-centered)
        u_σ = pm.HalfNormal("u_σ", sigma=0.5)
        u_z = pm.Normal("u_z", mu=0, sigma=1, dims="j")
        u = pm.Deterministic("u", u_z * u_σ, dims="j")

        # Random slopes (non-centered)
        w_σ = pm.HalfNormal("w_σ", sigma=0.5)
        w_z = pm.Normal("w_z", mu=0, sigma=1, dims="j")
        w = pm.Deterministic("w", w_z * w_σ, dims="j")

        # Linear predictor (per observation)
        μ = pm.Deterministic("μ", α + u[j] + (β + w[j]) * x, dims="i")

        # Likelihood
        pm.Bernoulli("y", p=pm.math.sigmoid(μ), observed=y, dims="i")

        # Sample
        idata = pm.sample(**kwargs)
        pm.compute_log_likelihood(idata)

    return model, idata


def main(data: pl.DataFrame, **kwargs):
    _, idata_freq = fit_model_freq(data, **kwargs)
    _, idata_nofreq = fit_model_nofreq(data, **kwargs)

    xr.DataArray(idata_freq).to_netcdf(utils.RESULTS_PATH / "post-freq.nc")
    xr.DataArray(idata_nofreq).to_netcdf(utils.RESULTS_PATH / "post-nofreq.nc")
    loos = az_stats.compare(
        {"freq": idata_freq, "nofreq": idata_nofreq}, method="BB-pseudo-BMA"
    )
    loos.to_parquet(utils.RESULTS_PATH / "loos.parquet")
    return idata_freq, idata_nofreq, loos


if __name__ == "__main__":
    syll = pl.read_parquet(utils.RESULTS_PATH / "syllables.parquet")
    idata_freq, idata_nofreq, loos = main(syll, target_accept=0.9, chains=4)
