# Import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def method():
    """
    Fit a linear regression model on the training data.

    Returns:
        output (float): The mean squared error of the model on the training data.
    """

    # Create a sample dataset
    np.random.seed(0)
    data = {
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Target': np.random.rand(100)
    }
    df = pd.DataFrame(data)

    # Split the data into features (X) and target (y)
    X = df[['Feature1', 'Feature2']]
    y = df['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the mean squared error of the model on the test set
    mse = mean_squared_error(y_test, y_pred)

    return mse

# Call the method for validation
output = method()
print("Mean Squared Error:", output)