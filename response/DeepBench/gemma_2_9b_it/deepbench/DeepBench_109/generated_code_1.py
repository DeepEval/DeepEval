from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsGraph

def create_knn_graph_and_index(features, n_neighbors=5, metric='euclidean', correct_duplicates=True, **kwargs):
    if correct_duplicates:
        features = np.unique(features, axis=0)
    knn_graph = KNeighborsGraph(n_neighbors=n_neighbors, metric=metric, **kwargs)
    knn_graph.fit(features)
    adj_matrix = csr_matrix(knn_graph.to_scipy())
    return adj_matrix, knn_graph

if __name__ == "__main__":
    features = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 2], [2, 3]])
    adj_matrix, knn_graph = create_knn_graph_and_index(features)
    print("Adjacency Matrix:\n", adj_matrix)
    print("KNN Graph:\n", knn_graph)