from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def method():
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Assume the best hyperparameters are already determined
    # For example, let's say the best hyperparameters for RandomForest were found to be:
    best_params = {'n_estimators': 100, 'max_depth': 3, 'random_state': 42}
    
    # Create the model with the best hyperparameters
    model = RandomForestClassifier(**best_params)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the accuracy as the output
    output = accuracy
    return output

# Call the method and print the output for validation
model_accuracy = method()
print(f"Model Accuracy with Best Hyperparameters: {model_accuracy:.2f}")