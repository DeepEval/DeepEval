from typing import Callable, Union
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

def pairwise_distances_chunked(
    X: np.ndarray,
    Y: np.ndarray,
    reduce_func: Union[None, Callable] = None,
    metric: str = "euclidean",
    n_jobs: int = 1,
    working_memory: int = 100,
) -> np.ndarray:
    distances = cdist(X, Y, metric)
    chunk_size, num_chunks = (working_memory, -(-distances.size // working_memory))  # ceil div
    for i in range(0, distances.size, chunk_size):
        chunk = distances[i:i + chunk_size].copy()
        if n_jobs > 1:
            chunk = Parallel(n_jobs=n_jobs)(delayed(np.nanmin) if reduce_func is not None else lambda x: x
                                             for _ in range(chunk.shape[0]))
            chunk = np.array(chunk)
        else:
            if reduce_func is not None:
                chunk = reduce_func(chunk)
        yield chunk
    if reduce_func is not None:
        yield reduce_func(distances)

if __name__ == "__main__":
    # Create sample input values
    X = np.random.rand(100, 5)
    Y = np.random.rand(50, 5)

    # Define a reduce function to calculate the mean
    def reduce_func(arr):
        return np.nanmean(arr)

    # Call the function and print the results
    for i, chunk in enumerate(pairwise_distances_chunked(X, Y, reduce_func, metric="euclidean", n_jobs=2)):
        print(f"Chunk {i}: {chunk}")