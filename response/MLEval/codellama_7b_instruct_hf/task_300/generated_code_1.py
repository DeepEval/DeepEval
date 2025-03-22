import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def method(data):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42)

    # Train a logistic regression model on the training set
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = model.accuracy(X_test, y_test)

    # Return the final output
    return {"accuracy": accuracy, "y_pred": y_pred}

# Example usage
data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10], "target": [0, 1, 0, 1, 0]})
output = method(data)
print(output)