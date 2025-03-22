from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

def method():
    # Create a synthetic dataset for demonstration purposes
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Validate the model on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.2f}")
    
    # Return the trained model
    return model

# Call the method function for validation
trained_model = method()