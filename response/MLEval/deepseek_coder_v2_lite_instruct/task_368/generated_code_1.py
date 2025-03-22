import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def method():
    # Sample data creation (replace this with your actual data)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.rand(100)      # 100 target values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the RandomForestRegressor model
    rf_model = RandomForestRegressor(random_state=42)

    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters found by grid search
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Create a new model with the best parameters
    best_rf_model = RandomForestRegressor(**best_params, random_state=42)
    best_rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_rf_model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Return the predictions
    output = y_pred
    return output

# Call the method for validation
if __name__ == "__main__":
    output = method()
    print("Predictions:", output)