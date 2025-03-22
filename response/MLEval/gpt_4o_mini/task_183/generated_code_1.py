import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

def method():
    # Load a sample dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Extra Trees Classifier
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Output the accuracy score
    return accuracy

# Call the method for validation
output = method()
print(f'Accuracy of Extra Trees Classifier: {output:.2f}')