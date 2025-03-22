import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Load dataset (for demonstration, we can use sklearn's built-in dataset)
    from sklearn.datasets import load_iris
    data = load_iris()
    
    # Create a DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Split the data into features and target
    X = df[data.feature_names]
    y = df['target']

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=200)

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy as output
    output = accuracy
    return output

# Call the method for validation
result = method()
print(f"Model accuracy: {result:.2f}")