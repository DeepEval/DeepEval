import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def create_knn_graph_and_index(features, n_neighbors=5, metric='minkowski', correct_duplicates=False, **kwargs):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    nbrs.fit(features)

    distances, indices = nbrs.kneighbors(features)
    
    if correct_duplicates:
        for i in range(features.shape[0]):
            unique_indices, unique_distances = np.unique(indices[i], return_index=True)
            distances[i] = distances[i][unique_distances]
            indices[i] = unique_indices
            
    n_samples = features.shape[0]
    row_ind = np.repeat(np.arange(n_samples), n_neighbors)
    col_ind = indices.flatten()
    data = distances.flatten()

    adjacency_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_samples, n_samples))
    
    return adjacency_matrix, nbrs

if __name__ == "__main__":
    sample_features = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    adjacency_matrix, knn_index = create_knn_graph_and_index(sample_features, n_neighbors=2, metric='euclidean')
    
    print("Adjacency matrix:")
    print(adjacency_matrix.toarray())
    
    print("\nKNN Index (distances and indices for the first sample):")
    distances, indices = knn_index.kneighbors(sample_features[:1])
    print("Distances:", distances)
    print("Indices:", indices)