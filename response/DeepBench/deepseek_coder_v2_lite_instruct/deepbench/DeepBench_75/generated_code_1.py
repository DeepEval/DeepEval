import numpy as np
from sklearn.neighbors import NearestNeighbors

def nearest_neighbors(X, n_neighbors, metric='euclidean', **kwargs):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs).fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices, distances, None  # rp_forest is not supported in this implementation

if __name__ == "__main__":
    # Example usage
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    n_neighbors = 2
    metric = 'euclidean'
    
    indices, distances, rp_forest = nearest_neighbors(X, n_neighbors, metric)
    
    print("Indices of nearest neighbors:\n", indices)
    print("Distances to nearest neighbors:\n", distances)