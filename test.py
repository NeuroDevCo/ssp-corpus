import numpy as np
from matplotlib import pyplot as plt


def sim_data():
    n_pop = np.array([5, 20, 10, 5, 5])
    ssp_pop = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    ssp_true = sum(ssp_pop * n_pop) / sum(n_pop)

    N = 1_000
    p_obs = np.array([0.4, 0.35, 0.3, 0.25, 0.2])

    sylls = np.random.choice(range(5), N, replace=True, p=p_obs / p_obs.sum())
    _, counts = np.unique(sylls, return_counts=True)
    ssp_group = [np.random.binomial(1, p, size=n) for p, n in zip(ssp_pop, counts)]
    ssp = np.concatenate(ssp_group)
    ssp_obs = ssp.mean()

    ssp_group_mean = [values.mean() for values in ssp_group]
    ssp_strat = sum(ssp_group_mean * n_pop) / n_pop.sum()

    return ssp_true, ssp_obs, ssp_strat


ssp_true, ssp_obs, ssp_strat = list(zip(*[sim_data() for _ in range(100)]))
fig, ax = plt.subplots(1, 1)
ax.hist(ssp_obs, color="lightgrey", bins=10, density=True, histtype="stepfilled")
ax.hist(ssp_strat, color="darkslategrey", bins=10, density=True, histtype="stepfilled")

ax.axvline(ssp_true[0], color="tab:red", linewidth=2)
ax.text(
    x=ssp_true[0] + (ssp_true[0] * 0.01),
    y=1,
    s=f"$\\theta$={ssp_true[0]:.2f}",
    rotation=90,
    ha="right",
    va="bottom",
    color="tab:red",
    size=14,
)
