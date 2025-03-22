import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def method():
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Define the model
    model = RandomForestClassifier()

    # Define the parameters to search over
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best parameters and the corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Train the model with the best parameters
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X, y)

    # Make predictions
    y_pred = best_model.predict(X)

    # Calculate the final output (e.g., accuracy score)
    output = accuracy_score(y, y_pred)

    return output

# Call the method for validation
output = method()
print("Final Output:", output)