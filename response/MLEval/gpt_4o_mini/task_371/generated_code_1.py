# Import necessary packages
import numpy as np
from sklearn.linear_model import LinearRegression

def method():
    # Generate some sample data
    X = np.array([[1], [2], [3], [4], [5]])  # Features
    y = np.array([1, 2, 3, 4, 5])            # Targets

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Return the output (predictions in this case)
    return predictions

# Call the method for validation
output = method()
print(output)