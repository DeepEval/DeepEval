import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Assuming you have your training data in a pandas DataFrame called 'X_train' and 'y_train'

def method():
    
    # Define the model
    model = LogisticRegression()

    # Define the parameter grid to search over
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Return the best estimator found
    return grid_search.best_estimator_

# Call the method and print the best estimator
output = method()
print("Best Estimator:", output)