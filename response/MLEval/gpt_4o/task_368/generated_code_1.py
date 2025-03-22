from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def method():
    # Sample data - You should replace these with your actual dataset
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.rand(100)     # 100 target values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor model
    rf = RandomForestRegressor(random_state=42)

    # Define a grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # The best model from the grid search
    best_rf = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_rf.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Return the best parameters and mean squared error
    output = {
        'best_params': grid_search.best_params_,
        'mse': mse
    }

    return output

# Call the method for validation
output = method()
print("Best Parameters:", output['best_params'])
print("Mean Squared Error:", output['mse'])