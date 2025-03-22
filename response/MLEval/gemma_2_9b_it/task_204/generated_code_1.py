import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Generate synthetic data with some non-linearity
    X, y = make_classification(n_samples=1000, n_features=2, kernel='poly', degree=2, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Support Vector Machine classifier with a polynomial kernel of degree 2
    svm = SVC(kernel='poly', degree=2)

    # Train the SVM model on the training data
    svm.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with polynomial kernel (degree 2): {accuracy:.2f}")

    return accuracy

# Call the method and store the output
output = method()