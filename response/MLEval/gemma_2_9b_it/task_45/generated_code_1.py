from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris

def method():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Define the hyperparameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs']
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Print the best parameters
    print("Best parameters:", grid_search.best_params_)
    # Print the best score
    print("Best score:", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_

# Call the method and store the result
best_model = method()

# You can now use best_model for predictions