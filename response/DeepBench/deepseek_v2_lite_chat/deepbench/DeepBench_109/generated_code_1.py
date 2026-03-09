import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def create_knn_graph_and_index(feature_array, n_neighbors=5, metric='euclidean', correct_duplicates=True, **search_index_args):
    # Create a KNN model
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **search_index_args)
    knn_model.fit(feature_array)
    
    # Get the distances and indices of the k-nearest neighbors
    distances, indices = knn_model.kneighbors(n_neighbors=n_neighbors)
    
    # Create a sparse adjacency matrix
    adjacency_matrix = sp.coo_matrix((np.ones_like(indices), (indices.ravel(), distances.ravel())),
                                    shape=(feature_array.shape[0], feature_array.shape[0]))
    
    # Correct for exact duplicates if requested
    if correct_duplicates:
        adjacency_matrix = adjacency_matrix.multiply(adjacency_matrix > 0)
    
    return adjacency_matrix, knn_model

if __name__ == "__main__":
    # Sample input values
    feature_array = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    
    # Call the function and print the results
    adjacency_matrix, knn_model = create_knn_graph_and_index(feature_array)
    print("Adjacency Matrix:\n", adjacency_matrix.todense())
    print("KNN Model:\n", knn_model)