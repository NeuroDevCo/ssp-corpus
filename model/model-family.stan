data {
  int<lower=1> N; // number of observations
  int<lower=1> J; // number of groups (random intercept levels)
  array[N] int<lower=1, upper=J> g; // group index for each observation
  array[N] int<lower=1> n; // number of trials per observation
  array[N] int<lower=0> y; // number of successes
}
parameters {
  real mu; // global intercept of the linear component (logit scale)
  real<lower=0> sigma; // standard deviation of random intercepts
  vector[J] r; // random intercepts for each group
  real<lower=0> phi; // precision (dispersion) parameter
}
transformed parameters {
  // Global proportion of success
  real<lower=0, upper=1> theta = inv_logit(mu);
  
  // Observation-specific alpha and beta for Beta-Binomial
  vector[N] alpha_obs;
  vector[N] beta_obs;
  
  for (i in 1 : N) {
    // Calculate group-level probability
    // The linear predictor is mu + r[group[i]]
    real p_group = inv_logit(mu + r[g[i]]);
    
    // Convert probability (p) and precision (phi) to alpha and beta
    // alpha = p * phi
    // beta  = (1 - p) * phi
    alpha_obs[i] = p_group * phi;
    beta_obs[i] = (1 - p_group) * phi;
  }
}
model {
  // Weakly informative priors
  mu ~ normal(0, 2);
  sigma ~ normal(0, 1); // Prior for random effect SD
  r ~ normal(0, sigma); // Random effects distribution
  phi ~ normal(0, 5); // Prior for precision (similar scale to original alpha/beta)
  
  // Likelihood
  for (i in 1 : N) {
    y[i] ~ beta_binomial(n[i], alpha_obs[i], beta_obs[i]);
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N] theta_rep;
  
  // Generate predictions for every observation
  for (i in 1 : N) {
    y_rep[i] = beta_binomial_rng(n[i], alpha_obs[i], beta_obs[i]);
    theta_rep[i] = y_rep[i] / n[i];
    log_lik[i] = beta_binomial_lpmf(y[i] | n[i], alpha_obs[i], beta_obs[i]);
  }
}
