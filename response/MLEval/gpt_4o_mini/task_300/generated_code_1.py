import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def method():
    # Sample data creation (for demonstration purposes)
    # In a real scenario, you would load your data here (e.g., from a CSV file)
    data = {
        'feature1': [0.5, 1.5, 2.5, 3.5, 4.5],
        'feature2': [1, 0, 1, 0, 1],
        'target': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Splitting the data into features and target
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating and fitting the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Making predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Returning the accuracy as the output
    output = accuracy
    return output

# Call the method for validation
result = method()
print(f'Model Accuracy: {result}')