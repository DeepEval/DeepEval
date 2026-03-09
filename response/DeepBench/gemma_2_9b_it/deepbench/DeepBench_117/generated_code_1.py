from joblib import Parallel, delayed
import numpy as np

def pairwise_distances_chunked(X, Y, reduce_func=None, metric='euclidean', n_jobs=1, working_memory=1000, **kwargs):
  if X.shape[0] == 0 or Y.shape[0] == 0:
    return np.zeros((0, 0))
  n_samples_X = X.shape[0]
  n_samples_Y = Y.shape[0]
  chunk_size = min(working_memory, n_samples_X, n_samples_Y)
  for i in range(0, n_samples_X, chunk_size):
    for j in range(0, n_samples_Y, chunk_size):
      chunk_X = X[i:i + chunk_size]
      chunk_Y = Y[j:j + chunk_size]
      distances = np.zeros((chunk_X.shape[0], chunk_Y.shape[0]))
      for k in range(chunk_X.shape[0]):
        for l in range(chunk_Y.shape[0]):
          distances[k, l] = np.linalg.norm(chunk_X[k] - chunk_Y[l], ord=2)
      if reduce_func is not None:
        distances = reduce_func(distances)
      yield distances
  

if __name__ == "__main__":
  X = np.random.rand(100, 10)
  Y = np.random.rand(100, 10)
  for chunk in pairwise_distances_chunked(X, Y, metric='manhattan', n_jobs=4, working_memory=20):
    print(chunk.shape)