from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():

    # Load the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train and fit the model on the training set
    model.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy
    print("Accuracy:", accuracy)

    # Return the accuracy
    return accuracy

# Call the method
accuracy = method()