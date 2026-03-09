import numpy as np

def _initialize_nmf(X, n_components, init='random', eps=1e-6, random_state=None):
    """
    Computes an initial guess for the non-negative rank k matrix approximation for a given data matrix X using Non-negative Matrix Factorization (NMF) initialization algorithms.

    Args:
        X: array-like of shape (n_samples, n_features)
            The data matrix to be decomposed.
        n_components: int
            The number of components desired in the approximation.
        init: {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
            Method used to initialize the procedure.
        eps: float, default=1e-6
            Truncate all values less than this in the output to zero.
        random_state: int, RandomState instance or None, default=None
            Used for reproducible results.

    Returns:
        W: array-like of shape (n_samples, n_components)
            Initial guesses for solving X ~= WH.
        H: array-like of shape (n_components, n_features)
            Initial guesses for solving X ~= WH.
    """

    # Random initialization
    if init == 'random':
        rng = np.random.RandomState(random_state)
        W = rng.rand(X.shape[0], n_components)
        H = rng.rand(n_components, X.shape[1])
    # NNDSVD initialization
    elif init == 'nndsvd':
        from sklearn.decomposition import NNDSVD
        svd = NNDSVD(n_components=n_components, random_state=random_state)
        svd.fit(X)
        W = svd.components_
        H = svd.components_
    # NNDSVD initialization with double centering
    elif init == 'nndsvda':
        from sklearn.decomposition import NNDSVD
        svd = NNDSVD(n_components=n_components, random_state=random_state)
        svd.fit(X - X.mean(axis=0))
        W = svd.components_
        H = svd.components_
    # NNDSVD initialization with double centering and average row adjustment
    elif init == 'nndsvdar':
        from sklearn.decomposition import NNDSVD
        svd = NNDSVD(n_components=n_components, random_state=random_state)
        svd.fit(X - X.mean(axis=0))
        W = svd.components_
        H = svd.components_
        H = H + (X.mean(axis=0) - X.mean(axis=0).dot(H)) / (H.sum(axis=1) + eps)
    # Custom initialization
    else:
        raise ValueError("Invalid initialization method.")

    # Truncate values less than eps to zero
    W[W < eps] = 0
    H[H < eps] = 0

    return W, H


if __name__ == "__main__":
    # Sample input values
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n_components = 2

    # Call the function and print the results
    W, H = _initialize_nmf(X, n_components)
    print(W)
    print(H)