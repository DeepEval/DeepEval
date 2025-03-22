import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def method():
    # Generate a random dataset for clustering
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Create a KMeans clustering model
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plotting the clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.title('KMeans Clustering')
    plt.show()

    # Return the cluster labels for validation
    return labels

# Call the method for validation
output = method()
print("Cluster Labels:", output)