import numpy as np
from scipy.sparse import linalg

def orthogonal_mp_gram(X, y, n_nonzero_coefs=None, tol=None, precompute=False, copy_X=True, return_path=False, return_n_iter=False):
    # Handle edge cases
    if n_nonzero_coefs is not None and n_nonzero_coefs > X.shape[1]:
        raise ValueError("Number of non-zero coefficients exceeds number of features.")

    # Precompute Gram matrix if necessary
    if precompute:
        gram_matrix = linalg.gram_matrix(X, copy=copy_X)
    else:
        gram_matrix = None

    # Initialize coefficients and residual
    coefficients = np.zeros(X.shape[1])
    residual = y - X @ coefficients

    # Iterate until convergence or maximum iterations reached
    n_iter = 0
    while (n_nonzero_coefs is None or n_iter < n_nonzero_coefs) and (tol is None or np.linalg.norm(residual) > tol):
        # Compute current solution
        current_solution = X.T @ residual

        # Compute Gram matrix
        if gram_matrix is None:
            gram_matrix = linalg.gram_matrix(X, copy=copy_X)

        # Compute residual
        residual = y - X @ current_solution

        # Compute coefficients
        coefficients = current_solution

        # Update iteration count
        n_iter += 1

    # Return solution and number of iterations if requested
    if return_path:
        return coefficients, n_iter
    elif return_n_iter:
        return coefficients, n_iter
    else:
        return coefficients

# Run example
if __name__ == "__main__":
    # Generate sample input data
    n_samples, n_features = 100, 1000
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)

    # Call function
    coefficients = orthogonal_mp_gram(X, y, n_nonzero_coefs=10, tol=1e-6, precompute=True, copy_X=False, return_path=False, return_n_iter=False)

    # Print results
    print(coefficients)
    print(np.linalg.norm(y - X @ coefficients))