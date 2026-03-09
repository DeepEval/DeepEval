import numpy as np
from sklearn.utils import check_random_state
from sklearn.decomposition import NMF

def _initialize_nmf(X, n_components, init='nndsvd', eps=1e-6, random_state=None):
    """
    Compute an initial guess for the non-negative rank k matrix approximation for a given data matrix X
    using Non-negative Matrix Factorization (NMF) initialization algorithms.
    
    Parameters:
    X (array-like of shape (n_samples, n_features)): The data matrix to be decomposed.
    n_components (int): The number of components desired in the approximation.
    init (str): Method used to initialize the procedure ('random', 'nndsvd', 'nndsvda', 'nndsvdar').
    eps (float): Truncate all values less than this in the output to zero.
    random_state (int, RandomState instance or None): Used for reproducible results.
    
    Returns:
    W (array-like of shape (n_samples, n_components)): Initial guesses for solving X ~= WH.
    H (array-like of shape (n_components, n_features)): Initial guesses for solving X ~= WH.
    """
    if init not in ['random', 'nndsvd', 'nndsvda', 'nndsvdar']:
        raise ValueError("Invalid initialization method. Choose from 'random', 'nndsvd', 'nndsvda', 'nndsvdar'.")
    
    if init == 'random':
        random_state = check_random_state(random_state)
        W = random_state.rand(X.shape[0], n_components) * eps
        H = random_state.rand(n_components, X.shape[1]) * eps
    elif init == 'nndsvd':
        nndsvd = NMF(n_components=n_components, init='nndsvd', random_state=random_state, sparse_reconstruction=True, max_iter=50)
        W, H = nndsvd.fit(X)
    elif init == 'nndsvda':
        nndsvd = NMF(n_components=n_components, init='nndsvda', random_state=random_state, sparse_reconstruction=True, max_iter=50)
        W, H = nndsvd.fit(X)
    elif init == 'nndsvdar':
        nndsvd = NMF(n_components=n_components, init='nndsvdar', random_state=random_state, sparse_reconstruction=True, max_iter=50)
        W, H = nndsvd.fit(X)
    
    return W, H

if __name__ == "__main__":
    # Sample input values
    X = np.random.rand(10, 5)  # Random 10x5 matrix as the data matrix
    n_components = 3
    init = 'nndsvd'
    eps = 1e-4
    
    # Call the function and print the results
    W, H = _initialize_nmf(X, n_components, init, eps)
    print("W:", W)
    print("H:", H)