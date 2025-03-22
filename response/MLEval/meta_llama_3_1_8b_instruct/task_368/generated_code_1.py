# Import necessary packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd

# Method to create, fit, tune and predict using RandomForestRegression model
def method():
    # Load Boston housing dataset
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['PRICE'] = boston.target

    # Define features and target
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [100, 200, 300],
       'max_depth': [None, 5, 10],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 5, 10]
    }

    # Initialize the model
    model = RandomForestRegressor()

    # Perform grid search to tune the hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best parameters
    print("Best Parameters:", best_params)

    # Make predictions using the best model
    y_pred = best_model.predict(X_test)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Return the output
    return {
        'best_model': best_model,
        'best_params': best_params,
       'mse': mse,
        'y_pred': y_pred
    }

# Call the method for validation
output = method()
print(output)