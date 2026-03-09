import numpy as np
from sklearn.neighbors import NearestNeighbors

def nearest_neighbors(X, n_neighbors, metric='euclidean', metric_params=None, use_angular_rp_tree=False,
                     random_state=None, low_memory=False, verbose=False):
    """
    Computes the k-nearest neighbors for each data point in X.

    Args:
        X: Input data.
        n_neighbors: Number of nearest neighbors to compute.
        metric: Metric to use for distance computation.
        metric_params: Arguments for the metric computation function.
        use_angular_rp_tree: Whether to use angular random projection trees for approximate nearest neighbor search.
        random_state: Random state for approximate computations.
        low_memory: Whether to use low memory mode for approximate nearest neighbor search.
        verbose: Whether to print status data during computation.

    Returns:
        knn_indices: Indices of the k-nearest neighbors.
        knn_dists: Distances to the k-nearest neighbors.
        rp_forest: Random projection forest used for searching (or None if not used).
    """

    # Create a NearestNeighbors object
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params=metric_params,
                          use_angular_rp_tree=use_angular_rp_tree, random_state=random_state,
                          low_memory=low_memory)

    # Fit the object to the data
    knn.fit(X)

    # Get the k-nearest neighbors
    knn_indices, knn_dists = knn.kneighbors(X)

    # Return the results
    return knn_indices, knn_dists, knn.random_state_

if __name__ == "__main__":
    # Create sample input data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

    # Call the nearest_neighbors function
    knn_indices, knn_dists, rp_forest = nearest_neighbors(X, n_neighbors=3)

    # Print the results
    print("Nearest Neighbors Indices:")
    print(knn_indices)

    print("Nearest Neighbors Distances:")
    print(knn_dists)