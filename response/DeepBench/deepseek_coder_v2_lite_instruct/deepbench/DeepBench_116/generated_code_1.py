import numpy as np
from numpy.linalg import norm

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute='auto', copy_X=True, return_path=False, return_n_iter=False):
    X = np.array(X)
    y = np.array(y)

    if copy_X:
        X = X.copy()

    n_samples, n_features = X.shape

    if n_nonzero_coefs is None:
        n_nonzero_coefs = max(10, int(0.1 * n_features))

    if tol is not None:
        n_nonzero_coefs = None

    if precompute == 'auto':
        precompute = n_samples > n_features

    if precompute:
        gram = np.dot(X.T, X)
    else:
        gram = None

    if y.ndim == 1:
        n_targets = 1
        y = y.reshape(-1, 1)
    else:
        n_targets = y.shape[1]

    if n_nonzero_coefs > n_features:
        raise ValueError("Number of non-zero coefficients exceeds the number of features.")

    coef = np.zeros((n_features, n_targets))
    residual = y.copy()
    support = np.zeros(n_features, dtype=bool)
    n_iter = []
    coef_path = []

    for t in range(n_nonzero_coefs):
        if precompute:
            projection = np.dot(X[:, ~support], np.dot(gram[~support][:, support], coef[support]))
        else:
            projection = np.dot(X[:, support], coef[support])

        inner_products = np.dot(X[:, ~support], (residual - projection).T)
        max_idx = np.argmax(np.abs(inner_products))
        max_val = np.abs(inner_products[max_idx])

        if tol is not None and max_val < tol:
            break

        support[~support][max_idx] = True
        coef_t = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
        coef[:, 0] = 0
        coef[support, 0] = coef_t

        residual = y - np.dot(X[:, support], coef_t)
        n_iter.append(t + 1)
        coef_path.append(coef[:, 0].copy())

    if return_path:
        coef_path = np.array(coef_path)

    if return_n_iter:
        n_iter = np.array(n_iter)

    if n_targets == 1:
        return coef[:, 0]
    else:
        return coef

if __name__ == "__main__":
    X = np.random.randn(100, 50)
    X[:, 0] = X[:, 1] + X[:, 2]  # Make the first feature sparse
    y = np.dot(X, np.array([1, -1, 1, 0, 0])[:, None]) + np.random.randn(100, 1) * 0.1

    coef = orthogonal_mp_gram(X, y, n_nonzero_coefs=3, return_n_iter=True)
    print("Coefficients:", coef)