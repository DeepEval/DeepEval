# Import necessary packages
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

def method():
    """
    This function compares the performance of KNN model with 1 and 5 neighbors.
    """
    
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Initialize the KNN model with 1 and 5 neighbors
    k1_model = KNeighborsClassifier(n_neighbors=1)
    k5_model = KNeighborsClassifier(n_neighbors=5)
    
    # Initialize the cross-validation object
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize lists to store the accuracy scores
    k1_accuracies = []
    k5_accuracies = []
    
    # Perform cross-validation for the KNN model with 1 and 5 neighbors
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the KNN model with 1 and 5 neighbors
        k1_model.fit(X_train, y_train)
        k5_model.fit(X_train, y_train)
        
        # Make predictions using the KNN model with 1 and 5 neighbors
        y_pred_k1 = k1_model.predict(X_test)
        y_pred_k5 = k5_model.predict(X_test)
        
        # Calculate the accuracy scores
        k1_accuracies.append(accuracy_score(y_test, y_pred_k1))
        k5_accuracies.append(accuracy_score(y_test, y_pred_k5))
    
    # Calculate the mean accuracy scores
    mean_k1_accuracy = np.mean(k1_accuracies)
    mean_k5_accuracy = np.mean(k5_accuracies)
    
    # Print the mean accuracy scores
    print(f"Mean cross-validated accuracy with 1 neighbor: {mean_k1_accuracy:.2f}")
    print(f"Mean cross-validated accuracy with 5 neighbors: {mean_k5_accuracy:.2f}")
    
    # Return the mean accuracy scores
    return mean_k1_accuracy, mean_k5_accuracy

# Call the method for validation
mean_k1_accuracy, mean_k5_accuracy = method()