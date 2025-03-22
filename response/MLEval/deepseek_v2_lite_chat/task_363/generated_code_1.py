from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifier (in this example, we use Support Vector Machines)
clf = SVC()

# Define the grid of hyperparameters to search over
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Return the best parameters found by GridSearchCV
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters found by GridSearchCV:", best_params)

# Return the best estimator found by GridSearchCV
best_clf = grid_search.best_estimator_

# Return the GridSearchCV object fitted to the training data
output = grid_search

# Call the method for validation
print("Validation of best model:")
grid_search.score(X_test, y_test)

return output