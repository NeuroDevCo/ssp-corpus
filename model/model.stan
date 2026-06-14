data {
  int<lower=1> N;
  array[N] int<lower=1> n;
  array[N] int<lower=0, upper=n> y;
}

parameters {
  real mu;
  real<lower=1e-8> sigma;  // Strictly positive
  real gamma;
  vector[N] u;
}

model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 10);      // Half-normal prior
  gamma ~ normal(0, 10);
  u ~ normal(0, sigma);

  for (j in 1:N) {
    y[j] ~ binomial_logit(n[j], mu + u[j] + gamma / n[j]);
  }
}

generated quantities {
  real theta = inv_logit(mu);
  array[N] real theta_j;
  for (j in 1:N) {
    theta_j[j] = inv_logit(mu + u[j]);
  }
}