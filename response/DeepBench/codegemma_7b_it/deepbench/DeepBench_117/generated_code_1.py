from joblib import Parallel, delayed
from numpy import asarray, empty, fromiter, ceil
from functools import partial
from itertools import product

def pairwise_distances_chunked(X, Y, reduce_func=None, metric="euclidean", n_jobs=1, working_memory=1e8, **kwargs):
    """
    Generates a distance matrix chunk by chunk with optional reduction.

    Args:
        X: Array-like of shape (n_samples_x, n_features).
        Y: Array-like of shape (n_samples_y, n_features).
        reduce_func: Function to apply to each chunk (default: None).
        metric: Metric to use for distance calculation (default: "euclidean").
        n_jobs: Number of parallel jobs to use (default: 1).
        working_memory: Maximum memory to use for intermediate computations (default: 1e8).

    Returns:
        Yields a contiguous slice of the distance matrix, optionally processed by reduce_func.
    """

    X = asarray(X)
    Y = asarray(Y)

    chunk_size = int(min(working_memory, X.nbytes * X.shape[0] // 2))

    for i in range(0, X.shape[0], chunk_size):
        for j in range(i, min(i + chunk_size, X.shape[0])):
            chunk = empty((Y.shape[0],))
            for k in range(Y.shape[0]):
                chunk[k] = distance(X[j], Y[k], metric, **kwargs)
            if reduce_func is not None:
                chunk = reduce_func(chunk)
            yield chunk

def distance(x, y, metric, **kwargs):
    """Calculates the distance between two points."""
    if metric == "euclidean":
        return ((x - y) ** 2).sum() ** 0.5
    elif metric == "manhattan":
        return abs(x - y).sum()
    elif metric == "cosine":
        return 1 - (x @ y) / ((x ** 2).sum() ** 0.5 * (y ** 2).sum() ** 0.5)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

if __name__ == "__main__":
    # Example usage:
    X = asarray([[1, 2], [3, 4], [5, 6]])
    Y = asarray([[2, 2], [4, 5], [6, 6]])

    for chunk in pairwise_distances_chunked(X, Y, reduce_func=lambda x: x.sum()):
        print(chunk)