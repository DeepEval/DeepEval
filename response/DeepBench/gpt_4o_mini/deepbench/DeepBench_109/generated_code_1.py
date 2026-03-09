import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def create_knn_graph_and_index(features, n_neighbors=5, metric='euclidean', correct_duplicates=False, **kwargs):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    if correct_duplicates:
        # Remove duplicate neighbors (keep the nearest ones)
        unique_indices = []
        unique_distances = []
        for i in range(indices.shape[0]):
            unique, counts = np.unique(indices[i], return_counts=True)
            min_index = np.argsort(counts)[-n_neighbors:]  # Select the top K with least duplicates
            unique_indices.append(unique[min_index])
            unique_distances.append(distances[i][min_index])
        indices = np.array(unique_indices)
        distances = np.array(unique_distances)

    row_indices = np.repeat(np.arange(features.shape[0]), n_neighbors)
    col_indices = indices.flatten()
    weights = distances.flatten()

    adjacency_matrix = csr_matrix((weights, (row_indices, col_indices)), shape=(features.shape[0], features.shape[0]))
    
    return adjacency_matrix, nbrs

if __name__ == "__main__":
    features = np.array([[1, 2], [2, 3], [3, 1], [5, 4]])
    adjacency_matrix, knn_index = create_knn_graph_and_index(features, n_neighbors=2, metric='euclidean', correct_duplicates=True)
    
    print("Adjacency Matrix:\n", adjacency_matrix.toarray())
    print("KNN Index:", knn_index)