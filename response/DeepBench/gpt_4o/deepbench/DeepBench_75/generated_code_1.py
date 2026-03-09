import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection

def nearest_neighbors(X, n_neighbors=5, metric='euclidean', metric_args=None,
                      use_angular_rp_trees=False, random_state=None,
                      low_memory=False, verbose=False):
    if metric_args is None:
        metric_args = {}

    if use_angular_rp_trees:
        rp_forest = [SparseRandomProjection(n_components='auto', random_state=random_state)
                     for _ in range(n_neighbors)]
        if verbose:
            print("Using angular random projection trees.")
        
        transformed_data = [rp.fit_transform(X) for rp in rp_forest]
        X = np.hstack(transformed_data)
    else:
        rp_forest = None

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **metric_args)
    nbrs.fit(X)

    knn_dists, knn_indices = nbrs.kneighbors(X)

    if verbose:
        print(f"Found nearest neighbors for {X.shape[0]} points using {metric} metric.")

    return knn_indices, knn_dists, rp_forest

if __name__ == "__main__":
    X_sample = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    n_neighbors = 2
    metric = 'euclidean'
    verbose = True

    indices, dists, rp_forest = nearest_neighbors(X_sample, n_neighbors, metric, verbose=verbose)

    print("Indices of nearest neighbors:")
    print(indices)
    print("Distances to nearest neighbors:")
    print(dists)
    print("Random Projection Forest:")
    print(rp_forest)