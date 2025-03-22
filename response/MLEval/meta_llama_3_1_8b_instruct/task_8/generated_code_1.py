# Import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np

def method():
    # Create a sample dataset
    np.random.seed(0)
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.where(np.random.rand(100) < 0.5, 0, 1)
    }
    df = pd.DataFrame(data)

    # Split the dataset into features and target
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Return the output
    return {
       'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred
    }

# Call the method for validation
output = method()
print(output)