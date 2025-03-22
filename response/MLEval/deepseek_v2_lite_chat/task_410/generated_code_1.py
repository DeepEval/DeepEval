from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def method():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Define the model
    model = SVC(random_state=42)

    # Define the grid of hyperparameters to search
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict on the testing set
    y_pred = best_model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)

    # Return the accuracy score
    return accuracy

# Call the method for validation
output = method()
print("Model Accuracy:", output)