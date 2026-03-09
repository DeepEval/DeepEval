import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def create_knn_graph_and_index(feature_array, num_neighbors=5, distance_metric='euclidean', correct_duplicates=False, **kwargs):
    # Create a nearest neighbors search object
    nn = NearestNeighbors(num_neighbors, distance_metric=distance_metric, **kwargs)

    # Fit the search object to the input feature array
    nn.fit(feature_array)

    # Get the nearest neighbors for each feature
    distances, indices = nn.kneighbors(feature_array)

    # Create a sparse, weighted adjacency matrix representing the KNN graph
    adjacency_matrix = csr_matrix((distances, indices), shape=(feature_array.shape[0], feature_array.shape[0]))

    # Correct for exact duplicates if necessary
    if correct_duplicates:
        adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])

    return adjacency_matrix, nn

if __name__ == "__main__":
    # Create sample input values
    feature_array = np.random.rand(100, 10)

    # Call the function and print the results
    adjacency_matrix, nn = create_knn_graph_and_index(feature_array, num_neighbors=5, distance_metric='euclidean', correct_duplicates=True)
    print(adjacency_matrix)
    print(nn)