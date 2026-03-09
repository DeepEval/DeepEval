import numpy as np

def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=True, normalize=False):
  n = len(X)
  kernel_values = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      kernel_values[i, j] = kernel(X[i], X[j])
  if rescale_class_labels:
    class_counts = np.bincount(Y)
    rescale_factor = 1.0 / class_counts
    Y = np.array(Y) * rescale_factor[Y]
  if normalize:
    kernel_values /= np.sum(kernel_values)
  return np.sum(Y[:, None] * kernel_values * Y[None, :])

if __name__ == "__main__":
  X = [1, 2, 3, 4, 5]
  Y = [1, 1, -1, -1, 1]
  
  def gaussian_kernel(x1, x2):
    return np.exp(-np.square(x1 - x2) / 0.1)

  polarity_value = polarity(X, Y, gaussian_kernel)
  print(f"Kernel polarity: {polarity_value}")