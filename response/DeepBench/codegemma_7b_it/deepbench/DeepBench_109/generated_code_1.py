import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree

def create_knn_graph_and_index(features, k=None, metric='euclidean', exact_duplicates=True, **kwargs):
    """Calculates the KNN graph from input features.

    Args:
        features: A 2D array of feature vectors.
        k: The number of nearest neighbors to consider.
        metric: The distance metric to use.
        exact_duplicates: Whether to correct exact duplicates in the feature array.
        kwargs: Additional keyword arguments for the search index constructor.

    Returns:
        A tuple containing a sparse, weighted adjacency matrix representing the KNN graph and a k-nearest neighbors search object fitted to the input feature array.
    """

    if k is None:
        k = len(features) - 1

    # Create a k-d tree from the feature array
    tree = KDTree(features, metric=metric, exact_duplicate=exact_duplicates)

    # Query the k-d tree to find the nearest neighbors for each point
    distances, indices = tree.query(features, k=k + 1)

    # Create a sparse adjacency matrix from the indices and distances
    row = np.repeat(np.arange(len(features)), k)
    col = indices.ravel()
    data = distances.ravel()
    graph = csr_matrix((data, (row, col)), shape=(len(features), len(features)))

    # Create a k-nearest neighbors search object
    index = tree

    return graph, index

if __name__ == "__main__":
    # Create sample input values
    features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # Call the function and print the results
    graph, index = create_knn_graph_and_index(features)
    print(graph)
    print(index)