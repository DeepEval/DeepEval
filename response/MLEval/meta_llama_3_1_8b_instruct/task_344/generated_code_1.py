import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.randn(100) * 2

# Create a PyMC3 model
def method():
    with pm.Model() as model:
        # Define priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Define the likelihood
        y_obs = pm.Normal('y_obs', mu=alpha + beta * X, sigma=sigma, observed=y)

        # Sample from the posterior
        step = pm.NUTS()
        trace = pm.sample(1000, step=step)

        # Return the posterior mean of the parameters
        return np.mean(trace, axis=0)

# Call the method for validation
output = method()
print(output)

# Plot the data and the regression line
plt.scatter(X, y)
plt.plot(X, output[0] + output[1] * X, 'r')
plt.show()