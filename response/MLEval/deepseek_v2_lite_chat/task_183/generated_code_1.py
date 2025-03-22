import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Load the iris dataset as an example
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Extra Trees classifier
    et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    et_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = et_classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the accuracy as output
    return accuracy

# Call the method for validation
output = method()
print("Accuracy of Extra Trees Classifier:", output)