data {
  int<lower=0, upper=1> prior;
  int<lower=1> N; // number of observations
  int<lower=1> J; // number of groups (languages)
  
  array[N] int<lower=0, upper=1> y; // binary response variable (0 or 1)
  array[N] real x; // predictor variable (standardized)
  array[N] int<lower=1, upper=J> j; // group index for each observation
}
parameters {
  vector[2] beta; // global intercept and slope of freq
  vector[J] mu; // group-level adjustments
  real<lower=0> sigma_mu; // SD of group-level adjustments
}
model {
  // prior
  beta ~ normal(0, 0.5);
  sigma_mu ~ exponential(10);
  mu ~ normal(0, sigma_mu); // hierarchical prior for group adjustments
  
  // likelihood
  if (prior == 0) {
    y ~ bernoulli_logit(beta[1] + mu[j] + (beta[2] * to_vector(x)));
  }
}
generated quantities {
  // Simulate replicated data (y_rep) based on PRIORS
  if (prior == 1) {
    array[N] int y_rep;
    
    for (n in 1 : N) {
      y_rep[n] = bernoulli_logit_rng(beta[1] + mu[j[n]] + beta[2] * x[n]);
    }
  }
}
