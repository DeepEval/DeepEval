import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances

def pairwise_distances_chunked(X, Y=None, reduce_func=None, metric='euclidean', n_jobs=-1, working_memory=1024, **kwargs):
    if Y is None:
        Y = X

    if metric == 'euclidean':
        dist_func = euclidean_distances
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    chunk_size = working_memory // (n_samples_Y * 4)  # Approximate size of each chunk

    def compute_chunk(start):
        end = min(start + chunk_size, n_samples_X)
        distances = dist_func(X[start:end], Y, **kwargs)
        if reduce_func:
            return reduce_func(distances)
        return distances

    results = Parallel(n_jobs=n_jobs)(delayed(compute_chunk)(i) for i in range(0, n_samples_X, chunk_size))

    if reduce_func:
        return results
    else:
        return np.vstack(results)

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.array([[1, 0], [0, 1], [2, 2]])
    distances = pairwise_distances_chunked(X, Y, metric='euclidean')
    print(distances)