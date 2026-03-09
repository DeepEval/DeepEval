import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial import distance

def create_knn_graph_and_index(features, n_neighbors=5, metric='minkowski', p=2, correct_duplicates=False, **kwargs):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto', metric=metric, p=p, **kwargs)
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)
    distances = distances[:, 1:]  # exclude self-distance
    indices = indices[:, 1:]  # exclude self-distance
    
    # Correct exact duplicates
    if correct_duplicates:
        # find duplicates
        duplicates = np.where(np.all(indices == indices[:, None], axis=2))[0]
        # set the weights to 0 for duplicates
        distances[duplicates, duplicates] = 0
    
    # build sparse adjacency matrix
    adj_matrix = csr_matrix((np.ones_like(indices).flatten(), (indices.flatten(), np.arange(indices.size))), shape=(features.shape[0], features.shape[0]))
    
    return adj_matrix, nbrs

if __name__ == "__main__":
    # create sample input values
    np.random.seed(0)
    features = np.random.rand(100, 10)
    
    # call the function
    adj_matrix, nbrs = create_knn_graph_and_index(features, n_neighbors=5, metric='euclidean', correct_duplicates=True)
    
    # print the results
    print(adj_matrix.toarray())
    print(nbrs.kneighbors_graph(features).toarray())