from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import numpy as np

def method(data, k=3, method='euclidean'):
    """
    Minimize the distance between the cluster and a data point in that cluster.
    
    Parameters:
    data (np.array): The data points, where each row is a point and each column is a feature.
    k (int): The number of clusters to find.
    method (str): The distance measure to use. Supported methods are 'euclidean', 'cityblock', 'cosine', and 'hamming'.
                    Defaults to 'euclidean'.
    
    Returns:
    np.array: The cluster centroids.
    """
    if method == 'euclidean':
        # Initialize the KMeans model with k clusters
        kmeans = KMeans(n_clusters=k)
        # Fit the model to the data
        kmeans.fit(data)
        # Calculate the distance from each data point to the cluster center
        distances = cdist(data, kmeans.cluster_centers_)
    elif method == 'cityblock':
        # Initialize the KMeans model with k clusters
        kmeans = KMeans(n_clusters=k)
        # Fit the model to the data
        kmeans.fit(data)
        # Calculate the distance from each data point to the cluster center
        distances = cdist(data, kmeans.cluster_centers_, metric='cityblock')
    elif method == 'cosine':
        # Initialize the KMeans model with k clusters
        kmeans = KMeans(n_clusters=k)
        # Fit the model to the data
        kmeans.fit(data)
        # Calculate the distance from each data point to the cluster center
        distances = cdist(data, kmeans.cluster_centers_, metric='cosine')
    elif method == 'hamming':
        # Initialize the KMeans model with k clusters
        kmeans = KMeans(n_clusters=k)
        # Fit the model to the data
        kmeans.fit(data)
        # Calculate the distance from each data point to the cluster center
        distances = cdist(data, kmeans.cluster_centers_, metric='hamming')
    else:
        raise ValueError("Unsupported distance method. Use 'euclidean', 'cityblock', 'cosine', or 'hamming'.")
    
    # Find the minimum distance for each point
    min_distances = np.argmin(distances, axis=1)
    
    # Update the cluster centers to the data points closest to them
    updated_centroids = data[min_distances]
    
    return updated_centroids

# Example usage
if __name__ == "__main__":
    # Example data
    data = np.array([
        [1, 2],
        [1, 4],
        [1, 0],
        [10, 2],
        [10, 4],
        [10, 0]
    ])
    
    # Number of clusters
    k = 3
    
    # Calculate and return the updated centroids
    output = method(data, k)
    print("Updated Centroids:", output)