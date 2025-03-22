# Import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def method():
    """
    Train a RandomForestClassifier model on the Iris dataset.

    Returns:
        output: The accuracy of the trained model.
    """

    # Load the Iris dataset
    iris = load_iris()

    # Split the dataset into features (X) and target (y)
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets (80% for training and 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the RandomForestClassifier class
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model using the training data
    clf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Call the method for validation
output = method()
print("Model Accuracy:", output)