import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def method():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Check if the accuracy is greater than 85%
    if accuracy > 0.85:
        output = f"Model accuracy: {accuracy:.2f} (Threshold met)"
    else:
        output = f"Model accuracy: {accuracy:.2f} (Threshold not met)"

    return output

# Call the method for validation
print(method())