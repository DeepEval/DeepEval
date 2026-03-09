import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

def pairwise_distances_chunked(X, Y=None, reduce_func=None, metric='euclidean', n_jobs=1, working_memory=10000, **kwargs):
    """
    Generates a distance matrix chunk by chunk with optional reduction.
    
    Parameters:
    X (np.array): First set of points. Shape (n_samples_X, n_features).
    Y (np.array): Second set of points. Shape (n_samples_Y, n_features). Default is None.
    reduce_func: Function to apply to each chunked distance matrix. Default is None.
    metric: Distance metric to use. Default is 'euclidean'.
    n_jobs: Number of parallel jobs to run for distance calculation. Default is 1.
    working_memory: Size of chunks in megabytes. Default is 10000.
    kwargs: Additional keyword arguments passed to scipy.spatial.distance.cdist.
    
    Returns:
    generator: Yields a contiguous slice of the distance matrix, optionally processed by reduce_func.
    """
    chunksize = working_memory * 1024 * 1024  # Convert megabytes to bytes
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    
    num_samples_X = X.shape[0]
    num_samples_Y = Y.shape[0]
    
    if num_samples_X < chunksize or num_samples_Y < chunksize:
        # If the datasets are small enough, calculate the entire distance matrix
        distances = cdist(X, Y, metric=metric, **kwargs)
    else:
        # Calculate chunks and apply reduce_func if provided
        for start in range(0, num_samples_X, chunksize):
            end = min(start + chunksize, num_samples_X)
            yield cdist(X[start:end], Y, metric=metric, **kwargs)
            if reduce_func is not None:
                distances = reduce_func(distances)
    
    if reduce_func is not None:
        distances = reduce_func(distances)
    
    return distances

if __name__ == "__main__":
    # Generate sample data
    random_state = 42
    X, Y = make_blobs(n_samples=1000, centers=3, random_state=random_state)
    
    # Example of using the function
    for chunk in pairwise_distances_chunked(X, Y, reduce_func=lambda x: x.mean(axis=0), metric='manhattan'):
        print(chunk)