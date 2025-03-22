import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Create a synthetic dataset for demonstration
    data = {
        'feature1': [3.5, 2.1, 1.3, 4.8, 2.9, 3.6, 1.7, 4.2, 3.1, 2.7],
        'feature2': [1.2, 2.3, 3.1, 1.4, 2.8, 1.9, 3.3, 1.6, 2.5, 3.0],
        'target': [0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)

    # Split the data into features and target
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Return the accuracy as the output
    output = accuracy

    return output

# Call the method for validation
output = method()
print(f"Model Accuracy: {output:.2f}")