from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def method():
    # Load a dataset (Iris dataset for simplicity)
    data = load_iris()
    X, y = data.data, data.target
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the Extra Trees Classifier
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Output the accuracy
    output = accuracy
    return output

# Call the method for validation
accuracy = method()
print(f"Accuracy of the Extra Trees model: {accuracy:.2f}")