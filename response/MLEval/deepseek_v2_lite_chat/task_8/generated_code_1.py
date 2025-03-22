from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def method():
    # Sample data (replace this with your actual dataset)
    data = {
        'feature1': [1, 0, 1, 0, 1],
        'feature2': [0, 1, 0, 1, 1],
        'target': [0, 0, 0, 1, 1]
    }
    df = pd.DataFrame(data)

    # Define features and target
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Return the accuracy score
    return accuracy

# Call the method for validation
output = method()
print("Model Accuracy:", output)