# Import necessary packages
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def method():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Use X_test and y_test as validation data
    # Predict using the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model on the test data
    accuracy = accuracy_score(y_test, y_pred)

    # Output could be the accuracy or the predictions, depending on what you want to return
    output = accuracy

    return output

# Call the method for validation
output = method()
print(f"Validation Accuracy: {output}")