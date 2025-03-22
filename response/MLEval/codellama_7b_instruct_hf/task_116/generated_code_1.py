import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def method(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regressor on the training data
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf.predict(X_test)

    # Calculate the mean of the predicted outcomes
    y_pred_mean = np.mean(y_pred)

    # Return the mean of the predicted outcomes
    return y_pred_mean