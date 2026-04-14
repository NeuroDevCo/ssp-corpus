data {
  int<lower=1> N; // number of observations
  int<lower=1> J; // number of groups (languages)
  
  array[N] int<lower=0, upper=1> y; // binary response variable (0 or 1)
  array[N] real x; // predictor variable (standardized)
  
  array[N] int<lower=1, upper=J> j; // group index for each observation
}
parameters {
  vector[2] beta; // global intercept and slope of freq
  vector[J] z_mu; // group-level adjustments, unconstrained, normal(0,1)
  real<lower=0> sigma_mu; // SD of group-level adjustments
}
transformed parameters {
  real theta = inv_logit(beta[1]);
  vector[J] mu = z_mu * sigma_mu;
  vector[J] theta_j = inv_logit(beta[1] + mu);
}
model {
  // prior
  beta ~ normal(0, 0.25);
  sigma_mu ~ exponential(2);
  z_mu ~ normal(0, 0.4);
  
  // likelihood
  y ~ bernoulli_logit((beta[1] + mu[j]) + (beta[2] * to_vector(x)));
}
