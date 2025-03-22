import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def method(X, y, alpha, l1_ratio):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model with the specified hyperparameters
    model = LinearRegression(alpha=alpha, l1_ratio=l1_ratio)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = np.mean((y_test - y_pred) ** 2)

    # Return the final output
    return mse


X = ... # input data
y = ... # target data
alpha = ... # regularization strength
l1_ratio = ... # L1 regularization ratio
output = method(X, y, alpha, l1_ratio)