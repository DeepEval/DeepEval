import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Generate synthetic data for demonstration purposes
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.squeeze() + np.random.randn(100) * 2

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Prepare the output
    output = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'mse': mse
    }

    return output

# Call the method for validation
output = method()
print(output)