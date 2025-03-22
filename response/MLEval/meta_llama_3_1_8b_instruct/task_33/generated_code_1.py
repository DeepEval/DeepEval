import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Generate sample data
np.random.seed(0)
data, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)


def method():
    # Initialize KMeans model with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    
    # Fit the model to the data
    kmeans.fit(data)
    
    # Get the cluster labels for each data point
    labels = kmeans.labels_
    
    # Get the cluster centroids
    centroids = kmeans.cluster_centers_
    
    # Calculate the minimum distance between the cluster and a data point in that cluster
    min_distances = np.min(np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
    
    return min_distances


# Call the method to get the minimum distances
min_distances = method()

# Print the minimum distances
print("Minimum distances:", min_distances)

# Plot the data points and the cluster centroids
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.show()