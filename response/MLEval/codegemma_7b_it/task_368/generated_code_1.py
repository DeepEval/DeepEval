from sklearn.ensemble import RandomForestRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

def method():
    # Generate sample data
    X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [2.5, 5.1, 7.8, 10.7, 13.9, 17.5, 22.0, 27.2, 34.1, 42.0]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the RandomForestRegression model
    model = RandomForestRegression()

    # Define the hyperparameter grid
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [None, 5, 10],
                  'min_samples_split': [2, 5, 10]}

    # Perform grid search cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model and predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Print the results
    print('Best parameters:', grid_search.best_params_)
    print('Mean squared error:', mse)

    # Return the output (optional)
    output = {'best_params': grid_search.best_params_, 'mse': mse}
    return output

# Call the method
method()