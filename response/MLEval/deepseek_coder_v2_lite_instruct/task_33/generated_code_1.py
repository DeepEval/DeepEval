import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def method():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Initialize KMeans with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=0)

    # Fit the KMeans model to the data
    kmeans.fit(X)

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Get the labels of the data points
    labels = kmeans.labels_

    # Calculate the sum of squared distances of data points to their closest cluster center
    distances = kmeans.transform(X)
    sum_of_squared_distances = np.sum(np.min(distances, axis=1))

    # The output can be the cluster centers or the sum of squared distances
    output = centers

    return output

# Call the method for validation
if __name__ == "__main__":
    centers = method()
    print("Cluster Centers:")
    print(centers)