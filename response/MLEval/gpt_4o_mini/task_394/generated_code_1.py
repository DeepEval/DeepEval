import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Generate some sample data
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Return the trained model and mean squared error as output
    output = {
        'model': model,
        'mean_squared_error': mse
    }
    return output

# Call the method for validation
result = method()
print("Trained Model:", result['model'])
print("Mean Squared Error on Test Set:", result['mean_squared_error'])