# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def method():
    # Create a sample dataset (replace with your actual dataset)
    np.random.seed(0)
    X = np.random.rand(100, 1)
    y = 3 * X + 2 + np.random.randn(100, 1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Decision Tree Regressor model with k=30
    model = DecisionTreeRegressor(max_depth=None, min_samples_split=2, random_state=42, max_features='auto', max_leaf_nodes=30)
    model.fit(X_train, y_train)

    # Predict the target variable using the trained model
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Print the MSE
    print("Mean Squared Error (MSE):", mse)

    # Plot the actual and predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

    # Return the MSE as the output
    return mse

# Call the method for validation
output = method()
print("Final Output:", output)