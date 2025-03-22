import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def method():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = SVC()
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    
    # Fit the model with the best parameters
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and estimator
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    # Output
    output = {
        "best_params": best_params,
        "classification_report": report
    }
    
    return output

# Call the method for validation
output = method()
print(output['best_params'])
print(output['classification_report'])