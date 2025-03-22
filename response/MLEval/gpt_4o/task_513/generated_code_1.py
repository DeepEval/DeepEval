# Import necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def method():
    # Load a sample dataset (for illustration purposes, we use the Iris dataset)
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # To make it a binary classification problem, we take only two classes
    X = X[y != 2]
    y = y[y != 2]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the Logistic Regression model
    model = LogisticRegression()
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Validate the model on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Prepare the output with the trained model and its accuracy
    output = {
        'model': model,
        'accuracy': accuracy
    }
    
    return output

# Call the method for validation
output = method()
print("Trained Logistic Regression Model:", output['model'])
print("Model Accuracy on Test Set:", output['accuracy'])