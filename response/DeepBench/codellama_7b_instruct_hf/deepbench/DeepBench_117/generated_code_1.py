import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_chunked

def pairwise_distances_chunked(X, Y, reduce_func, metric='euclidean', n_jobs=None, working_memory=None, **kwargs):
    # Calculate pairwise distances between rows in X and Y, chunk by chunk
    chunk_size = 10000
    n_chunks = int(np.ceil(X.shape[0] / chunk_size))
    n_rows_per_chunk = int(np.ceil(X.shape[0] / n_chunks))

    # Initialize output matrix
    dist_matrix = np.zeros((X.shape[0], Y.shape[0]))

    # Iterate over chunks
    for chunk_idx in range(n_chunks):
        # Get indices of rows in current chunk
        row_indices = np.arange(chunk_idx * n_rows_per_chunk, (chunk_idx + 1) * n_rows_per_chunk, dtype=int)

        # Calculate pairwise distances for current chunk
        dist_chunk = pairwise_distances(X[row_indices], Y, metric=metric, n_jobs=n_jobs, **kwargs)

        # Apply reduction function to current chunk
        if reduce_func is not None:
            dist_chunk = reduce_func(dist_chunk)

        # Add current chunk to output matrix
        dist_matrix[row_indices, :] = dist_chunk

    return dist_matrix

if __name__ == "__main__":
    # Generate sample input data
    X = np.random.rand(1000, 10)
    Y = np.random.rand(500, 10)

    # Calculate pairwise distances between X and Y, chunk by chunk
    dist_matrix = pairwise_distances_chunked(X, Y, reduce_func=np.mean)

    # Print output
    print(dist_matrix)