import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def create_knn_graph_and_index(features, k=5, metric='euclidean', correct_duplicates=True, **kwargs):
    if correct_duplicates:
        # Remove exact duplicates
        features = np.unique(features, axis=0)
    
    # Fit KNN search index
    knn_index = NearestNeighbors(n_neighbors=k, metric=metric, **kwargs)
    knn_index.fit(features)
    
    # Get KNN graph
    distances, indices = knn_index.kneighbors(features)
    adjacency_matrix = sp.coo_matrix((distances.flatten(), (indices.flatten(), np.arange(indices.size))), shape=(features.shape[0], features.shape[0]))
    
    return adjacency_matrix, knn_index

if __name__ == "__main__":
    # Sample input values
    features = np.array([[1, 2], [3, 4], [1, 4], [7, 8], [5, 6], [7, 9]])
    k = 2
    metric = 'euclidean'
    correct_duplicates = True
    
    # Call the function and print the results
    adjacency_matrix, knn_index = create_knn_graph_and_index(features, k, metric, correct_duplicates)
    print("Adjacency Matrix:\n", adjacency_matrix.toarray())
    print("KNN Index:\n", knn_index)