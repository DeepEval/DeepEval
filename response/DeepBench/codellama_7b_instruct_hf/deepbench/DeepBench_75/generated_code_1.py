import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

def nearest_neighbors(X, n_neighbors, metric, args=None, use_angular_rp=False, random_state=None, low_memory=False, verbose=False):
    if metric == 'euclidean':
        dists = euclidean_distances(X)
    elif metric == 'cosine':
        dists = 1 - np.dot(X, X.T)
    else:
        raise ValueError(f"Metric {metric} is not supported")
    
    if use_angular_rp:
        from sklearn.neighbors import AngularRPForest
        rp_forest = AngularRPForest(random_state=random_state)
        rp_forest.fit(dists)
        knn_indices = rp_forest.query(dists, n_neighbors=n_neighbors, return_distance=False)
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, metric_params=args, n_jobs=-1)
        nn.fit(X)
        knn_indices = nn.kneighbors(X, return_distance=False)
    
    knn_dists = dists[np.arange(X.shape[0])[:, None], knn_indices]
    
    if verbose:
        print(f"Number of neighbors computed: {n_neighbors}")
    
    return knn_indices, knn_dists, rp_forest

if __name__ == "__main__":
    # Create some sample input data
    X = np.random.rand(100, 10)
    
    # Compute nearest neighbors for each data point
    knn_indices, knn_dists, rp_forest = nearest_neighbors(X, 5, 'euclidean')
    
    # Print the indices and distances of the 5 nearest neighbors for the first data point
    print(knn_indices[0])
    print(knn_dists[0])