# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def method():
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Output the model parameters and performance metric
    output = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'mean_squared_error': mse
    }
    
    return output

# Call the method for validation
output = method()
print(output)