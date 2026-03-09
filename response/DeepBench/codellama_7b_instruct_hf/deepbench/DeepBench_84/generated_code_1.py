import numpy as np
from sklearn.decomposition import NMF

def initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    if init == 'random':
        # Initialize W and H randomly
        W = np.random.rand(X.shape[0], n_components)
        H = np.random.rand(n_components, X.shape[1])
    elif init == 'nndsvd':
        # Use NNDSVD initialization
        nmf = NMF(n_components=n_components, init='nndsvd')
        W, H = nmf.fit_transform(X)
    elif init == 'nndsvda':
        # Use NNDSVD initialization with absolute values
        nmf = NMF(n_components=n_components, init='nndsvda')
        W, H = nmf.fit_transform(X)
    elif init == 'nndsvdar':
        # Use NNDSVD initialization with absolute values and random initialization
        nmf = NMF(n_components=n_components, init='nndsvdar')
        W, H = nmf.fit_transform(X)
    else:
        # Raise an error if the init parameter is not valid
        raise ValueError("Invalid initialization method specified")

    # Truncate all values less than eps to zero
    W[W < eps] = 0
    H[H < eps] = 0

    return W, H

if __name__ == "__main__":
    # Create sample input values
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n_components = 2
    init = 'random'
    eps = 1e-6
    random_state = None

    # Call the function and print the results
    W, H = initialize_nmf(X, n_components, init, eps, random_state)
    print(W)
    print(H)