// modelling the syllabic frequency of SSP-following and-violating syllables

// this model estimates the difference in syllabic frequency between SSP-following and SSP-violating syllables. We use a log-normal distribution with a location and a scale parameter (mu and sigma, respectively), and model both parameters using a multilevel linear regression model that estimates the effect of SSP-followingness via a regression coefficient (beta_1, pi_1) with language-specific adjustments (u_1 and w_1).

data {
  int<lower=0> N; // total number of observations (syllables)
  int<lower=0> J; // number of languages
  array[N] int<lower=1, upper=J> j; // index of each language
  vector[N] x; // +0.5 if SSP-following, -0.5 if SSP-violating
  vector[N] y; // Zipf-score of syllabic frequency
}

parameters {
  // blobal parameters
  real beta_0; // global intercept for mu
  real beta_1; // blobal coefficient for x in mu
  real pi_0; // global intercept for sigma
  real pi_1; // global coefficient for x in sigma

  // non-centered group-level raw parameters
  vector[J] raw_u0;// raw group intercept adjustments for mu
  vector[J] raw_u1; // raw group slope adjustments for mu
  vector[J] raw_w0; // raw group intercept adjustments for sigma
  vector[J] raw_w1; // raw group slope adjustments for sigma

  // hyperparameters for group-level adjustments
  real<lower=0> sigma_u0; // SD for u_0
  real<lower=0> sigma_u1; // SD for u_1
  real<lower=0> sigma_w0; // SD for w_0
  real<lower=0> sigma_w1; // SD for w_1
}

transformed parameters {
  // transform raw parameters to centered parameters
  vector[J] u_0 = sigma_u0 * raw_u0;
  vector[J] u_1 = sigma_u1 * raw_u1;
  vector[J] w_0 = sigma_w0 * raw_w0;
  vector[J] w_1 = sigma_w1 * raw_w1;
}

model {
  // priors for global parameters
  beta_0 ~ normal(0, 10);
  beta_1 ~ normal(0, 10);
  pi_0 ~ normal(0, 5);
  pi_1 ~ normal(0, 5);

  // priors for raw group-level parameters
  raw_u0 ~ normal(0, 1);
  raw_u1 ~ normal(0, 1);
  raw_w0 ~ normal(0, 1);
  raw_w1 ~ normal(0, 1);

  // hyperpriors for group-level standard deviations
  sigma_u0 ~ exponential(2);
  sigma_u1 ~ exponential(2);
  sigma_w0 ~ exponential(2);
  sigma_w1 ~ exponential(2);

  // vectorized likelihood
  vector[N] mu = beta_0 + u_0[j] + (beta_1 + u_1[j]) .* x;
  vector[N] sigma = exp(pi_0 + w_0[j] + (pi_1 + w_1[j]) .* x);

  y ~ lognormal(mu, sigma);
}
