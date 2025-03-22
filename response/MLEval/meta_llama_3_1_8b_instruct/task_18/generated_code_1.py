# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


# Load the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a linear regression model
model = LinearRegression()


# Train the model using the training data
def method():
    model.fit(X_train, y_train)
    return model


# Make predictions on the test data
def evaluate_model(model):
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    return mae, mse, rmse

# Call the method to train the model
trained_model = method()

# Evaluate the trained model
mae, mse, rmse = evaluate_model(trained_model)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)