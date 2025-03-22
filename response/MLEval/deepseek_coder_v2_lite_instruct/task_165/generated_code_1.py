import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def method():
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Define the model with 1 neighbor
    knn = KNeighborsClassifier(n_neighbors=1)

    # Perform cross-validation with 5 folds
    scores = cross_val_score(knn, X, y, cv=5)

    # Calculate the mean cross-validated accuracy
    mean_accuracy = np.mean(scores)

    # Print the results
    print(f"Cross-validated accuracy with 1 neighbor: {mean_accuracy:.2f}")

    # Return the final output
    output = f"Cross-validated accuracy with 1 neighbor: {mean_accuracy:.2f}"
    return output

# Call the method for validation
output = method()
print(output)