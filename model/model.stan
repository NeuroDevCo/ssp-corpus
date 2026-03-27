data {
  int<lower=1> N; // number of observations
  int<lower=1> J; // number of groups
  array[N] int<lower=0> n; // number of trials per observation
  array[N] int<lower=0> y; // number of successes per observation
  array[N] int<lower=0, upper=J> j; // group index for each observation
}
parameters {
  real alpha; // population-level intercept
  vector[J] alpha_j; // group-level deviations
  real<lower=0> sigma; // standard deviation of group-level deviations
}
model {
  alpha ~ normal(0, 1.5);
  sigma ~ exponential(1);
  alpha_j ~ normal(0, sigma);
  y ~ binomial(n, inv_logit(alpha + alpha_j[j]));
}
generated quantities {
  real<lower=0, upper=1> theta; // probability for a "typical" group
  vector<lower=0, upper=1>[J] theta_j; // shrunken estimates for each group
  
  theta = inv_logit(alpha);
  for (j_idx in 1 : J) {
    theta_j[j_idx] = inv_logit(alpha + alpha_j[j_idx]);
  }
}
