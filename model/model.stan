data {
	int<lower=1> N; // number of observations
	int<lower=1> J; // number of groups
	array[N] int<lower=0> n; // number of trials per observation
	array[N] int<lower=0> y; // number of successes, out of the n trials
	array[N] int<lower=1, upper=J> j; // index of group
}

parameters {
	vector[J] theta; // group-level intercept
	real mu_j; // probability across groups
	real<lower=0> sigma_j; // SD of group-level intercepts
}


model {
	y ~ binomial_logit(n, theta[j]);
	theta ~ normal(mu_j, sigma_j);
	mu_j ~ student_t(3, 0, 2.5);
	sigma_j ~ exponential(4);
}