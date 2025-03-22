from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.3)

# Create a dictionary of parameters to tune
params = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'newton-cg', 'sag'],
    'max_iter': [100, 200, 300]
}

# Create a logistic regression model
model = LogisticRegression()

# Perform grid search cross-validation
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best parameters
print('Best parameters:', best_params)

# Evaluate the best model on the test set
accuracy = best_model.score(X_test, y_test)
print('Test accuracy:', accuracy)