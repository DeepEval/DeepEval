import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def method():
    # Load dataset (using Iris dataset as an example)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the best model (RandomForestClassifier in this case)
    best_model = RandomForestClassifier(random_state=42)
    best_model.fit(X_train, y_train)

    # Define the grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, 
                               scoring='accuracy', cv=3, verbose=2, n_jobs=-1)

    # Fit the model with grid search
    grid_search.fit(X_train, y_train)

    # Output the best parameters and best score
    output = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_score': grid_search.score(X_test, y_test)
    }

    return output

# Call the method for validation
if __name__ == "__main__":
    result = method()
    print(result)