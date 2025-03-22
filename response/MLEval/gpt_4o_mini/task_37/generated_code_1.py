import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

def method():
    # Load a sample dataset (Iris dataset for demonstration)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model with best hyperparameters
    # Here we use hypothetical best hyperparameters for RandomForest
    best_n_estimators = 100  # Example hyperparameter
    best_max_depth = 3        # Example hyperparameter

    model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Create an output dictionary
    output = {
        "accuracy": accuracy,
        "classification_report": report
    }

    return output

# Call the method for validation
if __name__ == "__main__":
    result = method()
    print("Accuracy:", result["accuracy"])
    print("Classification Report:\n", result["classification_report"])