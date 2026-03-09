import numpy as np
from scipy.sparse.linalg import lsqr

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute=None, copy_X=True, return_path=False, return_n_iter=False):
  if copy_X:
    X = X.copy()
  n_samples, n_features = X.shape
  if n_nonzero_coefs is None:
    n_nonzero_coefs = int(0.1 * n_features)
  if tol is not None:
    n_nonzero_coefs = None

  if n_nonzero_coefs > n_features:
    raise ValueError("Number of non-zero coefficients cannot exceed the number of features.")

  if precompute is None:
    precompute = 'auto'
  if precompute == 'auto' and (n_samples > 1000 or n_features > 1000):
    precompute = True

  if precompute:
    Gram = np.dot(X.T, X)
  else:
    Gram = lambda i: np.dot(X.T[:, i], X[:, i])

  n_targets = y.shape[1] if y.ndim == 2 else 1
  coefs = np.zeros((n_targets, n_nonzero_coefs))
  residual = y
  selected_indices = []
  n_iter = 0

  while n_nonzero_coefs > 0 and np.linalg.norm(residual)**2 > tol:
    n_iter += 1
    best_index = np.argmax(np.abs(np.dot(Gram(selected_indices), residual)))
    selected_indices.append(best_index)
    if return_path:
      coefs[:, -1] = np.dot(X[:, best_index], residual) / Gram(best_index)
    else:
      coefs[:, -1] = np.dot(X[:, best_index], residual) / Gram(best_index)
    residual -= np.dot(X[:, best_index], coefs[:, -1])
    n_nonzero_coefs -= 1

  if return_n_iter:
    return coefs, n_iter
  return coefs

if __name__ == "__main__":
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    coefs, n_iter = orthogonal_mp_gram(X, y, n_nonzero_coefs=5, tol=1e-3, precompute=True)
    print("Coefficients:", coefs)
    print("Number of iterations:", n_iter)