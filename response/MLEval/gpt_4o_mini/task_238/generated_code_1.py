import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameters
    n_estimators = 100  # Number of trees in the forest
    max_depth = 10      # Maximum depth of the tree

    # Create a Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the accuracy as the output
    output = {
        'accuracy': accuracy,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    
    return output

# Call the method for validation
result = method()
print(result)