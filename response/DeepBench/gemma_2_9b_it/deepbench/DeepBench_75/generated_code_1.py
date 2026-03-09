import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import RPTree

def nearest_neighbors(X, n_neighbors, metric='euclidean', metric_params=None, use_rp_tree=False, random_state=None, low_memory=False, verbose=False):
    if use_rp_tree:
        rp_forest = RPTree(n_estimators=100, random_state=random_state)
        rp_forest.fit(X)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params=metric_params, algorithm='brute')
        knn.fit(rp_forest.transform(X))
        knn_indices, knn_dists = knn.kneighbors(rp_forest.transform(X))
        return knn_indices, knn_dists, rp_forest
    else:
        if metric == 'euclidean':
            tree = KDTree(X)
        elif metric == 'manhattan':
            tree = BallTree(X)
        else:
            tree = KDTree(X)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params=metric_params, algorithm='auto', n_jobs=-1)
        knn.fit(X)
        knn_indices, knn_dists = knn.kneighbors(X)
        return knn_indices, knn_dists, None

if __name__ == "__main__":
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    knn_indices, knn_dists, rp_forest = nearest_neighbors(X, n_neighbors=2, metric='euclidean', use_rp_tree=True)
    print("kNN indices:", knn_indices)
    print("kNN distances:", knn_dists)
    print("Random Projection Forest:", rp_forest)