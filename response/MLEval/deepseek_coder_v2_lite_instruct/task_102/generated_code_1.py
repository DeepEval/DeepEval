import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def method():
    # Example data
    data = {
        'X1': [1, 2, 3, 4, 5],
        'X2': [2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)

    # Define X and y
    X = df[['X2']]  # Independent variable
    y = df['X1']    # Dependent variable

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients
    intercept = model.intercept_
    coefficient = model.coef_[0]

    # Interpret the coefficients
    output = f"Intercept: {intercept}\nCoefficient: {coefficient}"

    return output

# Call the method for validation
print(method())