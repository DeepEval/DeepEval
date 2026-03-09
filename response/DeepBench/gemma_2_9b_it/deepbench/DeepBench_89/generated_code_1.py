import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

def dict_learning(X, n_components, alpha=0.1, max_iter=100, tol=1e-8, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
  n_samples, n_features = X.shape

  if dict_init is None:
    dict_init = np.random.randn(n_features, n_components)
  if code_init is None:
    code_init = np.random.randn(n_samples, n_components)

  if method == 'lars':
    lasso = Lasso(alpha=alpha, max_iter=method_max_iter, n_jobs=n_jobs, random_state=random_state)
  else:
    raise ValueError("Method not supported")

  dictionary = dict_init
  code = code_init

  errors = []
  for iter in range(max_iter):
    if verbose:
      print(f"Iteration: {iter+1}")

    # Update the dictionary
    if positive_dict:
      dictionary = np.maximum(dictionary, 0)
    X_proj = code @ dictionary.T
    residual = X - X_proj
    if method == 'lars':
      lasso.fit(X_proj.T, dictionary.T)
      dictionary = lasso.coef_

    # Update the code
    if positive_code:
      code = np.maximum(code, 0)
    code = np.linalg.lstsq(dictionary, X, rcond=None)[0]

    # Calculate the error
    error = np.linalg.norm(X - X_proj)
    errors.append(error)

    if verbose:
      print(f"Error: {error}")

    if error < tol:
      if verbose:
        print("Stopping condition met.")
      break

    if callback is not None:
      callback(iter, dictionary, code, error)

  if return_n_iter:
    return code, dictionary, errors, iter + 1
  else:
    return code, dictionary, errors


if __name__ == "__main__":
    X = np.random.rand(10, 5)
    n_components = 3
    code, dictionary, errors = dict_learning(X, n_components)
    print("Code:\n", code)
    print("Dictionary:\n", dictionary)
    print("Errors:\n", errors)