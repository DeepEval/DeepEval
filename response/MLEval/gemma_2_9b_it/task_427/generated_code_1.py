import pystan
import numpy as np

def method():
    # Define the Stan model code
    stan_code = """
    data {
      int<lower=0> N; // Number of data points
      vector[N] y; // Observed data
    }
    parameters {
      real mu; // Mean
      real<lower=0> sigma; // Standard deviation
    }
    model {
      mu ~ normal(0, 1);
      sigma ~ cauchy(0, 5);
      y ~ normal(mu, sigma);
    }
    """

    # Compile the Stan model
    model = pystan.StanModel(model_code=stan_code)

    # Prepare the data for the model
    data = {
        "N": 100,
        "y": np.random.normal(loc=0, scale=1, size=100)
    }

    # Fit the model using Variational Bayes (VB)
    fit = model.sampling(data=data, iter=2000, chains=4, seed=123, method="vb")

    # Extract posterior samples
    mu_samples = fit.extract()["mu"]
    sigma_samples = fit.extract()["sigma"]

    # Print summary statistics
    print("Posterior mean of mu:", np.mean(mu_samples))
    print("Posterior standard deviation of mu:", np.std(mu_samples))
    print("Posterior mean of sigma:", np.mean(sigma_samples))
    print("Posterior standard deviation of sigma:", np.std(sigma_samples))

    return mu_samples, sigma_samples


# Call the method to run the Stan model
mu_samples, sigma_samples = method()