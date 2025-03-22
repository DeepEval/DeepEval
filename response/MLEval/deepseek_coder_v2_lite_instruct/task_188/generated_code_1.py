import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def method():
    # Example data (replace with your actual data)
    X_train = np.array([[1], [2], [3], [4], [5]])
    Y_train = np.array([2, 4, 5, 4, 5])

    # Define the model
    model = LinearRegression()

    # Define the parameter grid for grid search
    param_grid = {}  # For linear regression, no parameters to tune

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Fit the best model on the entire training data
    best_model.fit(X_train, Y_train)

    # Example prediction (replace with your actual data)
    X_test = np.array([[6], [7], [8]])
    Y_pred = best_model.predict(X_test)

    # Prepare the output
    output = {
        'best_model': best_model,
        'Y_pred': Y_pred
    }

    return output

# Call the method for validation
result = method()
print(result)