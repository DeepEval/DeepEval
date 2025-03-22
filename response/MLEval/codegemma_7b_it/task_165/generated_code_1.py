from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def method():
    # Load the dataset
    X = ...  # Replace with your dataset
    y = ...  # Replace with your target variable

    # Create a KNeighborsClassifier with 1 neighbor
    knn = KNeighborsClassifier(n_neighbors=1)

    # Perform cross-validation
    scores = cross_val_score(knn, X, y, cv=5)

    # Calculate the mean cross-validated accuracy
    mean_accuracy = scores.mean()

    # Return the mean accuracy
    return mean_accuracy

# Call the method to get the mean accuracy
mean_accuracy = method()

# Print the mean accuracy
print("Mean cross-validated accuracy with 1 neighbor:", mean_accuracy)