import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def method(total_request_rate, cpu):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(total_request_rate, cpu, test_size=0.2, random_state=42)

    # Create a LinearRegression model and fit the data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = np.mean((y_test - y_pred) ** 2)

    # Print the results
    print("Mean squared error:", mse)

    # Return the model and the test data
    return model, X_test, y_test

# Generate some example data
total_request_rate = np.array([10, 20, 30, 40, 50])
cpu = np.array([10, 20, 30, 40, 50])

# Call the method and print the results
model, X_test, y_test = method(total_request_rate, cpu)
print("Predicted values:", y_pred)