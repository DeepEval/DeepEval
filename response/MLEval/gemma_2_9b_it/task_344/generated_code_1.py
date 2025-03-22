import pymc3 as pm
import numpy as np

def method():
    # Define the model
    with pm.Model() as model:
        # Priors
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        # Likelihood
        y = np.random.normal(mu, sigma, size=100)  # Example data
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        # Sample from the posterior
        trace = pm.sample(2000, tune=1000)
    
    # Extract posterior samples
    posterior_mu = trace["mu"]
    posterior_sigma = trace["sigma"]

    # Calculate and return desired output
    output = {"posterior_mu": posterior_mu.mean(), "posterior_sigma": posterior_sigma.mean()}

    return output

# Validate the method
result = method()
print(result)