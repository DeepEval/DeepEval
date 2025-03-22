from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def method():
    # Load a sample dataset, for example, the iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # For simplicity, let's assume a binary classification problem
    # Filter out only two classes
    X = X[y != 2]
    y = y[y != 2]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Return the trained model and the accuracy
    output = {
        'model': model,
        'accuracy': accuracy
    }

    return output

# Call the method for validation
result = method()
print("Trained Logistic Regression Model:", result['model'])
print("Accuracy on test data:", result['accuracy'])