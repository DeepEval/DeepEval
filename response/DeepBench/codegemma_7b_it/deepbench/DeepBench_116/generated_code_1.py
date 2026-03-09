import numpy as np

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute=True, copy_X=True, return_path=False, return_n_iter=False):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for solving sparse linear regression problems.

    Args:
        X: A 2D array-like of shape (n_samples, n_features) representing the input data. Columns should be assumed to have unit norm.
        y: A 1D or 2D array-like of shape (n_samples,) or (n_samples, n_targets) representing the target values.
        n_nonzero_coefs: An integer specifying the desired number of non-zero coefficients in the solution. If None, this defaults to 10% of the number of features.
        tol: A float specifying the maximum squared norm of the residual. If provided, this overrides n_nonzero_coefs.
        precompute: A boolean or the string 'auto' indicating whether to precompute the Gram matrix for faster computation when n_targets or n_samples is large.
        copy_X: A boolean indicating whether to copy the input matrix X (default is True).
        return_path: A boolean indicating whether to return the entire coefficient path for each iteration.
        return_n_iter: A boolean indicating whether to return the number of iterations taken for convergence.

    Returns:
        The coefficients of the OMP solution as a 1D or 2D NumPy array, depending on the input, with an option to return the number of iterations if return_n_iter is True.
        If return_path is True, also returns a list of coefficients for each iteration.
    """

    X = np.atleast_2d(X)
    y = np.atleast_1d(y)

    if copy_X:
        X = X.copy()

    n_samples, n_features = X.shape
    n_targets = y.shape[1] if y.ndim == 2 else 1

    if n_nonzero_coefs is None:
        n_nonzero_coefs = int(n_features / 10)

    if tol is None:
        tol = n_features / n_samples

    if precompute == 'auto':
        precompute = n_targets > 1 or n_samples > 256

    if precompute:
        G = np.dot(X.T, X)
    else:
        G = None

    coef = np.zeros((n_features, n_targets))
    residual = y

    for n_iter in range(n_nonzero_coefs):
        if np.linalg.norm(residual) <= tol:
            break

        if precompute:
            s = np.dot(X.T, residual)
        else:
            s = np.dot(X.T, X).dot(residual)

        # Find the index of the most significant coefficient
        j = np.argmax(np.abs(s))

        # Update the coefficients and residual
        coef[j] = s[j] / G[j, j]
        residual -= coef[j] * X[:, j]

    if return_n_iter:
        return coef, n_iter

    if return_path:
        return coef, [coef]

    return coef

if __name__ == "__main__":
    # Create sample input values
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 15, 20])

    # Call the OMP function
    coef, n_iter = orthogonal_mp_gram(X, y, return_n_iter=True)

    # Print the results
    print("Coefficients:", coef)
    print("Number of iterations:", n_iter)