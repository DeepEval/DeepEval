import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def method():
    # Generate some synthetic data
    np.random.seed(42)  # For reproducibility
    data = np.random.rand(100, 2)  # 100 points in 2D

    # Define the number of clusters
    n_clusters = 3

    # Create KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the model
    kmeans.fit(data)

    # Get the cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Calculate the distance between each point and its cluster center
    distances = np.linalg.norm(data - centers[labels], axis=1)

    # Minimize the distance (the distances should be as small as possible after fitting)
    output = {
        'centers': centers,
        'labels': labels,
        'distances': distances
    }

    return output

# Call the method for validation
output = method()

# Optional: Visualize the results
def visualize_clusters(data, output):
    plt.scatter(data[:, 0], data[:, 1], c=output['labels'], cmap='viridis', marker='o', label='Data Points')
    plt.scatter(output['centers'][:, 0], output['centers'][:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.show()

visualize_clusters(np.random.rand(100, 2), output)

# Print the output for verification
print("Cluster Centers:\n", output['centers'])
print("Labels:\n", output['labels'])
print("Distances to Centroids:\n", output['distances'])