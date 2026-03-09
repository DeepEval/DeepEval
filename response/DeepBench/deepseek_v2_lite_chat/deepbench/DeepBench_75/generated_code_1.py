import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors

def nearest_neighbors(X, n_neighbors, metric, *, low_memory=False, verbose=False):
    if metric == 'ball_tree':
        model = BallTree(X)
    elif metric == 'kd_tree':
        model = BallTree(X, leaf_size=30, metric='euclidean', sparse_output=True)
    else:
        model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, leaf_size=30)
        model.fit(X)

    knn_indices = model.kneighbors(n_neighbors=n_neighbors)[0]
    knn_dists = np.linalg.norm(X[knn_indices] - X, axis=1)
    
    if verbose:
        print('Computing nearest neighbors...')
        print('Finished computing nearest neighbors.')
        print(f'n_samples: {X.shape[0]}')
        print(f'n_neighbors: {n_neighbors}')
    
    return knn_indices, knn_dists, model

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 5)
    n_neighbors = 5
    metric = 'ball_tree'

    knn_indices, knn_dists, rp_forest = nearest_neighbors(X, n_neighbors, metric)
    
    print("KNN Indices:")
    print(knn_indices)
    print("\nKNN Dists:")
    print(knn_dists)