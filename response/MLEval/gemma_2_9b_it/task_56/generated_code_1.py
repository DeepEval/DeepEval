# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Load your data here (replace with your actual data loading)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]) 
    y = np.array([0, 1, 0, 1, 0])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy
    return accuracy

# Call the method and print the result
output = method()
print("Accuracy:", output)