import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Global variable to hold the model
model = None
X_train, X_test, y_train, y_test = None, None, None, None

def initialize_model():
    """Initialize the logistic regression model."""
    return LogisticRegression(max_iter=200)

def load_data():
    """Load the iris dataset and split it into training and test sets."""
    data = load_iris()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    """Train the logistic regression model."""
    global model, X_train, X_test, y_train, y_test
    model = initialize_model()
    X_train, X_test, y_train, y_test = load_data()
    model.fit(X_train, y_train)

def evaluate_model():
    """Evaluate the model on the test set."""
    global model, X_test, y_test
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def method():
    """Restart the model from the beginning and return the output."""
    train_model()  # Restart training
    output = evaluate_model()  # Get evaluation results
    return output

# Call the generated method for validation
if __name__ == "__main__":
    result = method()
    print(f"Model Accuracy: {result:.2f}")