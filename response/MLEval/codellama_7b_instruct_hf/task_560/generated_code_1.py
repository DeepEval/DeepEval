import numpy as np
from sklearn.cluster import KMeans

def method():
    # Generate some sample data
    X = np.random.rand(100, 10)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)

    # Get the cluster labels for the data
    labels = kmeans.labels_

    # Get the centroids for each cluster
    centroids = kmeans.cluster_centers_

    # Print the results
    print("Cluster labels:", labels)
    print("Centroids:", centroids)

    return centroids

# Call the method for validation
centroids = method()