# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Load the dataset (I used iris dataset for demonstration, replace it with your dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into features (X) and target (y)
# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define the model
def method():
    # Define the model
    model = LogisticRegression(max_iter=1000)

    # Define the hyperparameter tuning space
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_pca, y_train)

    # Train the best model on the entire dataset
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_pca, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test_pca)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy
    print(f"Accuracy: {accuracy:.2f}")

    # Return the output
    return accuracy

# Call the method for validation
output = method()
print(output)