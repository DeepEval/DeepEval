from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def method():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the parameter grid for the grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    # Create a SVM model
    svm = SVC()

    # Setup the grid search with cross-validation
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    # Fit the model with the best parameters
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Predict with the best model
    y_pred = grid_search.predict(X_test)

    # Generate a classification report
    report = classification_report(y_test, y_pred)

    # Print results
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-validation Score: {best_score}")
    print("Classification Report:")
    print(report)

    # Return the best parameters and the classification report as output
    output = {
        "best_params": best_params,
        "best_score": best_score,
        "classification_report": report
    }
    return output

# Call the method for validation
output = method()