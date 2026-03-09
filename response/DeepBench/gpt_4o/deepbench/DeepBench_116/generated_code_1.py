import numpy as np
from numpy.linalg import norm
from sklearn.utils.extmath import safe_sparse_dot

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute='auto', copy_X=True, return_path=False, return_n_iter=False):
    X = np.array(X, copy=copy_X)
    y = np.array(y, copy=False)

    n_samples, n_features = X.shape
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n_targets = y.shape[1]

    if n_nonzero_coefs is None and tol is None:
        n_nonzero_coefs = max(int(0.1 * n_features), 1)

    if n_nonzero_coefs is not None and n_nonzero_coefs > n_features:
        raise ValueError("Number of non-zero coefficients cannot exceed number of features.")

    if tol is not None:
        tol = tol ** 2

    if precompute == 'auto':
        precompute = n_targets > 1 or n_samples > n_features

    if precompute:
        G = safe_sparse_dot(X.T, X)
        Xy = safe_sparse_dot(X.T, y)

    coef = np.zeros((n_features, n_targets))
    residuals = y.copy()
    active_set = []
    selected_atoms = []
    errors = []

    for k in range(n_nonzero_coefs if tol is None else n_features):
        if precompute:
            correlations = safe_sparse_dot(G, residuals).sum(axis=1)
        else:
            correlations = np.dot(X.T, residuals).sum(axis=1)

        correlations[active_set] = 0
        best_atom = np.argmax(np.abs(correlations))
        active_set.append(best_atom)

        if precompute:
            subproblem = safe_sparse_dot(np.linalg.inv(G[np.ix_(active_set, active_set)]), Xy[active_set])
        else:
            subproblem = np.linalg.lstsq(X[:, active_set], y, rcond=None)[0]

        selected_atoms.append(subproblem)

        residuals = y - np.dot(X[:, active_set], subproblem)

        error = norm(residuals) ** 2
        errors.append(error)

        if tol is not None and error <= tol:
            break

    coef[active_set] = subproblem

    if y.ndim == 1:
        coef = coef.ravel()

    result = [coef]
    if return_path:
        result.append(selected_atoms)
    if return_n_iter:
        result.append(len(active_set))

    return tuple(result) if len(result) > 1 else result[0]

if __name__ == "__main__":
    X = np.random.rand(100, 20)
    X /= norm(X, axis=0)  # Normalize columns to have unit norm
    true_coef = np.zeros(20)
    true_coef[:5] = np.random.rand(5)
    y = np.dot(X, true_coef) + 0.01 * np.random.randn(100)

    coef = orthogonal_mp_gram(X, y, n_nonzero_coefs=5)
    print("Estimated coefficients:", coef)