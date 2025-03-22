import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def method():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree Classifier
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Check if accuracy is greater than 85%
    if accuracy > 0.85:
        output = f"Model accuracy: {accuracy * 100:.2f}% - Threshold met!"
    else:
        output = f"Model accuracy: {accuracy * 100:.2f}% - Threshold not met."

    return output

# Call the method for validation
result = method()
print(result)