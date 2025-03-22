from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Load the Iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Initialize the SVC model
    svc = svm.SVC()

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Fit the model with the best parameters
    best_svc = svm.SVC(**best_params)
    best_svc.fit(X_train, y_train)

    # Make predictions with the best model
    y_pred = best_svc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Return the best model for further use or analysis
    return best_svc

# Call the method for validation
output = method()