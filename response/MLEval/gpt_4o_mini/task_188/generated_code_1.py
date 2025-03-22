# Import necessary packages
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset creation (you should replace this with your actual dataset)
from sklearn.datasets import make_classification
X_train, Y_train = make_classification(n_samples=100, n_features=20, random_state=42)

def method():
    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, Y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Optionally, you can evaluate the model on validation data if available
    # Here we will just return the best model and its score
    best_score = grid_search.best_score_
    
    output = {
        'best_model': best_model,
        'best_score': best_score
    }
    
    return output

# Call the method for validation
result = method()
print(result)