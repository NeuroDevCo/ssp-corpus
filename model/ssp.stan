// modelling the prevalence of the SSP

// this model takes the observed number of SSP-following syllables in each language along with the total number of syllables in each language j, and produces a global estimate of SSP prevalence (theta) and a langugae specific estimate of SSP prevalence (theta_j)

data {
  int<lower=1> N; // total number of observations
  array[N] int<lower=1> n; // language-specific number of observations
  array[N] int<lower=0, upper=n> y; // number of observed successes in each language
}

parameters {
  real mu; // location parameter of theta
  vector[N] u; // language-specific adjustment to mu
  real<lower=0> sigma_u; // SD of language-specific adjustments
  real gamma; // bias parameter for number of observations
}

model {
  mu ~ std_normal();
  gamma ~ std_normal();
  u ~ normal(0, sigma_u);
  sigma_u ~ exponential(2);

  for (j in 1:N) {
    y[j] ~ binomial_logit(n[j], mu + u[j] + gamma / n[j]);
  }
}

generated quantities {
  real theta = inv_logit(mu); // global estimate of SSP-prevalence
  array[N] real theta_j; // language-specific estimate of SSP-prevalence
  
  for (j in 1:N) {
    theta_j[j] = inv_logit(mu + u[j]);
  }
}