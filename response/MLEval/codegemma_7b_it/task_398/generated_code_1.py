from sklearn.ensemble import RandomForestRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

# Define the method
def method():

    # Load the data (replace with your actual data loading code)
    X = ...
    y = ...

    # Create the RandomForestRegression model
    model = RandomForestRegression()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    # Perform grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    # Get the best model and its predictions
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X)

    # Calculate the mean squared error
    mean_squared_error = np.mean((predictions - y) ** 2)

    # Return the output (optional)
    output = {'mean_squared_error': mean_squared_error}
    return output

# Call the method for validation
result = method()

# Print the mean squared error
print('Mean Squared Error:', result['mean_squared_error'])