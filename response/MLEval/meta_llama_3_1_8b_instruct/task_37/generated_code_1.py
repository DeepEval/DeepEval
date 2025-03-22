# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the iris dataset
def load_data():
    iris = load_iris()
    return pd.DataFrame(data=iris.data, columns=iris.feature_names), pd.Series(iris.target)

# Define the method to test the model with best hyperparameters
def method():
    # Load the data
    X, y = load_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    model = RandomForestClassifier()

    # Define the hyperparameter tuning space
    param_grid = {
        'n_estimators': [100, 200, 300],
       'max_depth': [None, 5, 10],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best hyperparameters
    print("Best Hyperparameters:", best_params)

    # Evaluate the best model on the testing set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Testing Accuracy:", accuracy)

    # Return the output
    return accuracy

# Call the method for validation
output = method()
print("Output:", output)