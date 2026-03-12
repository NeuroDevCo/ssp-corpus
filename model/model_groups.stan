data {
  int<lower=1> N; // number of observations
  array[N] int<lower=1> n; // number of trials per observation
  array[N] int<lower=0> y; // number of successes
  
  // New data for random intercepts
  int<lower=1> J; // number of groups
  array[N] int<lower=1, upper=J> g; // group index for each observation
}
parameters {
  // Global parameters
  real intercept; // global intercept (log-odds scale)
  real<lower=0> kappa; // global dispersion (concentration)
  real<lower=0> sigma; // standard deviation of random intercepts
  
  // Non-centered random effects (efficient sampling)
  vector[J] z; // unscaled group effects
}
transformed parameters {
  // Group-level parameters
  vector<lower=0, upper=1>[J] mu_group; // group-specific probabilities
  vector<lower=0>[J] alpha_group; // group-specific alpha
  vector<lower=0>[J] beta_group; // group-specific beta
  
  for (j in 1 : J) {
    // 1. Convert intercept + random effect to probability (0 to 1)
    // sigma * z[j] is the deviation for group j
    mu_group[j] = inv_logit(intercept + sigma * z[j]);
    
    // 2. Convert probability and dispersion to alpha/beta
    // alpha = mu * kappa
    // beta = (1 - mu) * kappa
    alpha_group[j] = mu_group[j] * kappa;
    beta_group[j] = (1 - mu_group[j]) * kappa;
  }
}
model {
  // Priors
  intercept ~ normal(0, 1.5);
  kappa ~ normal(0, 5);
  sigma ~ exponential(1);
  z ~ std_normal(); // implies random effects ~ normal(0, sigma)
  
  // Likelihood
  for (i in 1 : N) {
    y[i] ~ beta_binomial(n[i], alpha_group[g[i]], beta_group[g[i]]);
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N] theta_rep;
  
  for (i in 1 : N) {
    // Posterior Predictive
    y_rep[i] = beta_binomial_rng(n[i], alpha_group[g[i]], beta_group[g[i]]);
    
    // Empirical probability of the replicated data
    theta_rep[i] = y_rep[i] / n[i];
    
    // Log-likelihood for LOO
    log_lik[i] = beta_binomial_lpmf(y[i] | n[i], alpha_group[g[i]],
                   beta_group[g[i]]);
  }
}
