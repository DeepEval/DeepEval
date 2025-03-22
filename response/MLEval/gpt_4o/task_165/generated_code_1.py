from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def method():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create KNeighborsClassifier with 1 neighbor
    knn = KNeighborsClassifier(n_neighbors=1)

    # Perform cross-validation and calculate mean accuracy
    cv_scores = cross_val_score(knn, X, y, cv=5)  # 5-fold cross-validation
    mean_accuracy = np.mean(cv_scores)

    # Output
    output = mean_accuracy
    return output

# Call the method and print the output for validation
print("Mean cross-validated accuracy with 1 neighbor:", method())