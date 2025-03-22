import pandas as pd
import numpy as np
import statsmodels.api as sm

def method():
    # Generate a sample dataset
    np.random.seed(0)  # For reproducibility
    X1 = np.random.rand(100) * 10  # Feature variable
    Y = 2.5 * X1 + np.random.randn(100) * 2  # Response variable with some noise

    # Prepare the data for regression
    X1 = sm.add_constant(X1)  # Adds a constant term for the intercept
    model = sm.OLS(Y, X1)  # Ordinary Least Squares regression
    results = model.fit()  # Fit the model

    # Get the coefficients
    intercept, slope = results.params

    # Output the coefficients and their interpretation
    output = {
        'intercept': intercept,
        'slope': slope,
        'interpretation': {
            'intercept': f"The intercept (DADMY) is {intercept:.2f}. It represents the expected value of Y when X1 is 0.",
            'slope': f"The slope (MOMMY) is {slope:.2f}. It indicates that for each unit increase in X1, the expected value of Y increases by {slope:.2f}."
        }
    }

    return output

# Validate the method
output = method()
print(output)