data {
  int<lower=1> N; // number of observations
  array[N] int<lower=1> n; // number of trials per observation
  array[N] int<lower=0> y; // number of successes
}
parameters {
  real<lower=0> alpha; // Alpha parameter of Beta distribution
  real<lower=0> beta; // Beta parameter of Beta distribution
}
transformed parameters {
  real mu = alpha / (alpha + beta);
  real phi = alpha + beta; // Precision parameter
  real theta = alpha / (alpha + beta); // Global proportion of success
}
model {
  // weakly informative priors 
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  y ~ beta_binomial(n, alpha, beta);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N] theta_rep;
  vector[N] y_prior;
  
  // Generate predictions for every observation
  for (i in 1 : N) {
    // beta_binomial_rng(trials, alpha, beta)
    y_rep[i] = beta_binomial_rng(n[i], alpha, beta);
    theta_rep[i] = y_rep[i] / n[i];
    log_lik[i] = beta_binomial_lpmf(y[i] | n[i], alpha, beta);
  }
}
