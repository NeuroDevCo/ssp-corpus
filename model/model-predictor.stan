data {
  int<lower=1> N; // number of observations
  array[N] int<lower=1> n; // number of trials per observation
  array[N] int<lower=0> y; // number of successes
  vector[N] X; // predictor variable
  int<lower=1> J; // number of groups
  array[N] int<lower=1, upper=J> j; // group index for each observation
}
parameters {
  real alpha; // population intercept
  real beta; // population slope for X
  real<lower=0> sigma_alpha; // standard deviation of random intercepts
  vector[J] a; // specific intercept for each group
}
transformed parameters {
  vector[N] mu = alpha + a[j] + beta * X;
  vector[N] theta = inv_logit(mu);
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma_alpha ~ exponential(1);
  a ~ normal(0, sigma_alpha);
  y ~ binomial_logit(n, mu);
}
