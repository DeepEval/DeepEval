from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

def method():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Create a KNN classifier with 1 neighbor
    knn = KNeighborsClassifier(n_neighbors=1)

    # Perform 5-fold cross-validation
    scores = cross_val_score(knn, X, y, cv=5)

    # Calculate the mean cross-validated accuracy
    mean_accuracy = scores.mean()

    # Print the mean accuracy
    print(f"Mean cross-validated accuracy with 1 neighbor: {mean_accuracy:.2f}")

    return mean_accuracy

# Call the method and store the output
output = method()