# Import necessary packages
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier()

# Define the parameters for grid search
param_grid = {
    'n_estimators': [10, 50, 100, 200],
   'max_depth': [None, 5, 10, 20],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 5, 10]
}

# Perform grid search and fit the model
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Define the method
def method():
    # Return the output
    return f"Best Parameters: {best_params}\nBest Model: {best_model}"

# Call the method for validation
print(method())