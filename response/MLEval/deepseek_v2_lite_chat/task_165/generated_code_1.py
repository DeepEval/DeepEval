from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

def method():
    # Generate a synthetic binary classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_clusters_per_class=1, random_state=42)
    
    # Define the range of k values to test
    k_values = range(1, 11)
    
    # Initialize a list to store the cross-validated accuracies
    accuracies = []
    
    # Loop over the k values
    for k in k_values:
        # Initialize the KNN classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Compute cross-validated accuracy
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        accuracy = np.mean(scores)
        accuracies.append(accuracy)
        
        # Print the accuracy for the current k value
        print(f"Accuracy with k={k}: {accuracy:.4f}")
    
    # Determine the k value with the highest accuracy
    best_k = k_values[np.argmax(accuracies)]
    print(f"\nBest accuracy is with k={best_k} (mean accuracy: {np.max(accuracies):.4f})")
    
    # Plot the accuracies against k values
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='--', markersize=10)
    plt.title('Accuracy vs. Number of Neighbors (K)')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    return best_k  # Return the best k value

# Call the method for validation
output = method()
print(f"\nBest k value for highest accuracy: {output}")