import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def method(X_train, y_train, X_test, y_test, best_params):
    """
    Trains and evaluates a Logistic Regression model using the best hyperparameters found.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        best_params: A dictionary containing the best hyperparameters.

    Returns:
        A string containing the classification report.
    """

    # Create a Logistic Regression model with the best hyperparameters
    model = LogisticRegression(**best_params)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate a classification report
    report = classification_report(y_test, y_pred)
    print(report)

    return report

# Example usage:
# Assuming you have already loaded your data and found the best hyperparameters
X_train, X_test, y_train, y_test = train_test_split(data['features'], data['target'], test_size=0.2, random_state=42)
best_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'} 

# Call the method to test the model
output = method(X_train, y_train, X_test, y_test, best_params)