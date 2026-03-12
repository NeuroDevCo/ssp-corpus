from src.model import family_id
from pathlib import Path

import arviz as az
import arviz_plots as azp
from cmdstanpy import CmdStanModel
from matplotlib import pyplot as plt
import numpy as np
import polars as pl

azp.style.use("arviz-variat")

RANDOM_SEED: int = 1234
rng = np.random.default_rng(RANDOM_SEED)


N: int = 1000
ALPHA = 0.7
ALPHA_SD = 0.5


def inv_logit(p: np.ndarray):
    return np.exp(p) / (1 + np.exp(p))


def create_data(
    n_obs: int = N,
    n_fam: int = N // 10,
    prop_mean: float = ALPHA,
    prop_sd: float = ALPHA_SD,
):
    lang = list(range(n_obs))
    fam_id = np.random.choice(range(n_fam), size=n_obs)
    n = np.random.randint(100, 5_000, size=n_obs)
    alpha = np.random.normal(ALPHA, ALPHA_SD, n_obs)
    y = np.random.binomial(n, inv_logit(alpha))
    return {"N": n_obs, "iso639p3code": lang, "family_id": fam_id, "n": n, "y": y}


if __name__ == "__main__":
    # data = create_data()
    # df = pl.DataFrame(data)

    df = pl.read_parquet(Path("out", "data.parquet")).join(
        pl.read_parquet(Path("out/languages.parquet")).select(
            ["iso639p3code", "family_id"]
        ),
        on="iso639p3code",
        how="inner",
    )
    families = {k: i + 1 for i, k in enumerate(set(df["family_id"]))}
    n_fam = len(set(df["family_id"]))

    data = {
        "N": len(df),
        "n": df["n"].to_list(),
        "y": df["sum"].to_list(),
        "prop": df["prop"].to_list(),
        "J": n_fam,
        "g": [families[f] for f in df["family_id"]],
    }
    stan_file = Path("model/model.stan")
    model = CmdStanModel(stan_file=stan_file)

    samples = model.sample(
        data=data,
        iter_sampling=1_000,
        iter_warmup=1_000,
        adapt_delta=0.9,
        seed=RANDOM_SEED,
        step_size=0.01,
        chains=8,
        parallel_chains=8,
    )

    idata = az.from_cmdstanpy(
        posterior=samples,
        posterior_predictive="theta_rep",
        observed_data={"y": data["prop"]},
        log_likelihood={"y": "log_lik"},
    )

    azp.plot_rank_dist(idata, var_names=["mu", "alpha", "beta"])
    az.plot_ppc(idata, data_pairs={"y": "theta_rep"}, num_pp_samples=25)

    stan_file = Path("model/model_groups.stan")
    model = CmdStanModel(stan_file=stan_file)

    samples_groups = model.sample(
        data=data,
        iter_sampling=1_000,
        iter_warmup=1_000,
        adapt_delta=0.95,
        seed=RANDOM_SEED,
        step_size=0.001,
        chains=8,
        parallel_chains=8,
    )

    idata_groups = az.from_cmdstanpy(
        posterior=samples_groups,
        posterior_predictive="theta_rep",
        observed_data={"y": data["prop"]},
        log_likelihood={"y": "log_lik"},
    )

    azp.plot_rank_dist(idata_groups, var_names=["kappa", "sigma", "intercept"])
    az.plot_ppc(idata_groups, data_pairs={"y": "theta_rep"}, num_pp_samples=25)

    cmp_df = az.compare(
        {"model": idata, "model_groups": idata_groups}, method="BB-pseudo-BMA"
    )
    az.plot_compare(cmp_df)
