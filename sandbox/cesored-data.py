from dataclasses import dataclass
from string import ascii_letters
from random import choices, sample

from matplotlib import pyplot as plt
import numpy as np
import polars as pl

n_lang = 100
n_obs_max = 100
b_n_obs_ssp = -1
global_ssp = 0.5


@dataclass
class Language:
    name: str
    n: int = 1_000
    p_ssp: float = global_ssp

    def __post_init__(self):
        sylls = ["".join(choices(ascii_letters, k=3)) for _ in range(self.n)]
        is_ssp = np.random.choice(
            [True, False], p=[self.p_ssp, 1 - self.p_ssp], size=self.n
        )
        data = {"name": self.name, "syll": sylls, "is_ssp": is_ssp}

        self.data = pl.DataFrame(data)


languages = []
for lang in range(n_lang):
    name = "".join(choices(ascii_letters, k=8))
    p_ssp = np.random.uniform(0, 1)
    prop_obs = np.random.uniform(1 / n_obs_max, 1)

    language = Language(name, n=n_obs_max, p_ssp=global_ssp)
    n = int(round(n_obs_max * prop_obs))
    language.data = language.data.sample(n)
    languages.append(language)


df = pl.concat([lang.data for lang in languages])
df_summary = df.group_by("name").agg(pl.col("is_ssp").mean(), pl.len().alias("n"))

fig, (ax_hist, ax_scat) = plt.subplots(1, 2)

ax_hist.axvline(
    global_ssp,
    color="orange",
    ls="dotted",
    lw=2,
    label="True SSP",
)
ax_hist.axvline(
    global_ssp,
    color="orange",
    lw=2,
    label="True SSP",
)
ax_hist.axvline(
    df_summary.get_column("is_ssp").mean(),
    color="darkslategray",
    lw=2,
    label="Observed SSP",
)
ax_hist.scatter(
    df_summary["is_ssp"],
    df_summary["name"],
    color="slategray",
    label="Observed SSP",
    alpha=1 / 2,
)
ax_hist.scatter(
    p_ssp, df_summary["name"], color="orange", label="True SSP", alpha=1 / 2
)

ax_hist.set_xlim(0, 1)
ax_hist.set_yticklabels([])
ax_hist.set_yticks([])

ax_scat.scatter(df_summary["n"], df_summary["is_ssp"], c="lightgrey", alpha=0.5)
coef = np.polyfit(df_summary["n"].to_list(), df_summary["is_ssp"].to_list(), 1)
fit = np.poly1d(coef)
ax_scat.plot(df_summary["n"], fit(df_summary["n"]), lw=2, color="darkslategrey")
ax_scat.set_ylim(0, 1)
ax_scat.set_ylabel("Proportion of SSP-following syllables")
ax_scat.set_xlabel("Number of observations")

fig.legend()
fig.tight_layout()


# censored data
n_lang = 192
n_syll = 1_000
data = []
for lang in range(n_lang):
    p_ssp = np.random.beta(1, 1)
    p_obs = np.random.uniform(0.1, 1.0)
    n_obs = int(n_syll * p_obs)
    ssp = np.random.binomial(1, p_ssp, n_syll)
    ssp_cens = ssp[np.random.choice(range(n_syll), size=n_obs, replace=False)]
    d = (n_obs, float(ssp.mean()), float(ssp_cens.mean()))
    data.append(d)

fig, (ax, ax_cens) = plt.subplots(1, 2)
n, ssp, ssp_cens = zip(*data)
ax.scatter(n, ssp_cens, c="grey", alpha=1 / 4)
coef = np.polyfit(n, ssp_cens, 1)
fit = np.poly1d(coef)
ax.plot(n, fit(n), lw=3, c="grey")
ax.set_title("Full data")

ax_cens.scatter(n, ssp, c="darkslategrey", alpha=1 / 4)
coef = np.polyfit(n, ssp, 1)
fit = np.poly1d(coef)
ax_cens.plot(n, fit(n), lw=3, c="darkslategrey")
ax_cens.set_title("Censored data")

# censored data
n_lang = 192
n_syll = 1_000
data = []
for lang in range(n_lang):
    p_ssp = np.random.beta(1, 1)
    prop_obs = np.random.uniform(0.1, 1)
    ssp = np.random.normal(p_ssp, 0.1, n_obs)
    p_obs = np.random.binomial(1, np.exp(p_ssp) / (1 + np.exp(p_ssp)), n_syll)
    n_obs = int(n_syll * prop_obs)
    ssp_cens = ssp_cens[np.int64(np.where(p_obs)[0])]
    d = (n_obs, float(ssp.mean()), float(ssp_cens.mean()))
    data.append(d)

n, ssp, ssp_cens = zip(*data)

coef = np.polyfit(n, ssp, 1)
fit = np.poly1d(coef)
preds = fit(n)

coef_cens = np.polyfit(n, ssp_cens, 1)
fit_cens = np.poly1d(coef_cens)
preds_cens = fit(n)

fig, axes = plt.subplots(2, 2)
axes[0, 0].scatter(n, ssp, c="lightgrey", alpha=1 / 4)
axes[0, 0].plot(n, preds, lw=3, c="grey")
axes[0, 0].hist(ssp)
axes[0, 0].set_ylim(0, 1)

axes[0, 1].hist(ssp_cens, color="lightgrey")
axes[0, 1].axvline(np.mean(ssp_cens), color="orange")
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_title(f"Mean = {np.mean(ssp):.2f}")

axes[1, 0].scatter(n, ssp_cens, c="darkslategrey", alpha=1 / 4)
axes[1, 0].plot(n, preds_cens, lw=3, c="darkslategrey")
axes[1, 0].set_ylim(0, 1)

axes[1, 1].hist(ssp_cens, color="darkslategrey")
axes[1, 1].axvline(np.mean(ssp_cens), color="orange")
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_title(f"Mean = {np.mean(ssp_cens):.2f}")

fig.supxlabel("N")


def inv_logit(x: float | np.ndarray):
    return np.exp(x) / (1 + np.exp(x))


def sim_theta(n_lang: int = 100):
    sigma_0 = np.random.exponential(10)
    mu_0j = np.random.normal(0, sigma_0, size=n_lang)
    beta_0 = np.random.normal(2, 0.5)

    return inv_logit(beta_0 + mu_0j)


thetas = sim_theta()

n = 1000
data = np.random.binomial(1, thetas, size=(1_000, len(thetas)))
p_missing = np.stack([np.arange(1_000) / 1_000] * 100)
obs = np.random.binomial(1, 1 - p_missing, size=(100, 1000))
y = data.sum(axis=0)
y_cens = np.ma.masked_array(data, mask=obs.T).sum(axis=0).data
prop = y / 1_000
prop_cens = y_cens / obs.sum(axis=1)

coef = np.polyfit([n] * 100, prop, 1)
fit = np.poly1d(coef)
preds = fit(n)

coef = np.polyfit(obs.sum(axis=1), prop_cens, 1)
fit = np.poly1d(coef)
preds_cens = fit(obs.sum(axis=1))


fig, (ax_hist, ax_scat) = plt.subplots(1, 2)
ax_hist.hist(prop, color="darkslategrey")
ax_hist.hist(prop_cens, color="lightgrey")
ax_hist.set_xlim(0, 1)

ax_scat.scatter(n, prop, color="darkslategrey", alpha=0.5)
ax_scat.scatter(n, prop_cens, color="orange", alpha=0.5)
ax_scat.plot(n, preds, lw=3, c="darkslategrey")
ax_scat.plot(obs.sum(axis=1), preds_cens, lw=3, c="orange")
ax_scat.set_ylim(0, 1)
