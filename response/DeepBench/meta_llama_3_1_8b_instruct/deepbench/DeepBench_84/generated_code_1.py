import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

def _initialize_nmf(X, n_components, init='random', eps=1e-6, random_state=None):
    if init == 'random':
        W = np.random.rand(X.shape[0], n_components)
        H = np.random.rand(n_components, X.shape[1])
        return W, H

    elif init == 'nndsvd':
        U, s, Vt = svds(X, k=n_components)
        s[s < eps] = 0
        U *= s[:, None]
        Vt *= s[None, :]
        W = Vt.T
        H = U.T
        return W, H

    elif init == 'nndsvda':
        U, s, Vt = svds(X, k=min(n_components, X.shape[0]), which='LM')
        s[s < eps] = 0
        U *= s[:, None]
        Vt *= s[None, :]
        tmp = np.dot(Vt.T, X)
        W = np.dot(tmp, Vt) / np.sum(tmp**2, axis=0)[:, None]
        H = np.dot(U, Vt.T) / np.sum(U**2, axis=1)[:, None]
        return W, H

    elif init == 'nndsvdar':
        U, s, Vt = svds(X, k=min(n_components, X.shape[0]), which='LM')
        s[s < eps] = 0
        U *= s[:, None]
        Vt *= s[None, :]
        W = Vt.T
        H = U.T
        return W, H

    else:
        raise ValueError("Invalid init parameter. Must be 'random', 'nndsvd', 'nndsvda', or 'nndsvdar'.")

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    X = np.random.rand(100, 10)

    # Initialize NMF
    n_components = 5
    init = 'nndsvd'
    W, H = _initialize_nmf(X, n_components, init=init)

    # Print results
    print("W:\n", W)
    print("\nH:\n", H)