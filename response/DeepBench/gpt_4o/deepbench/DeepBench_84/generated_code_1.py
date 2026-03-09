import numpy as np
from sklearn.decomposition import NMF
from sklearn.utils import check_random_state

def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    n_samples, n_features = X.shape
    rng = check_random_state(random_state)

    if init == 'random':
        W = rng.rand(n_samples, n_components)
        H = rng.rand(n_components, n_features)
        W[W < eps] = 0
        H[H < eps] = 0
        return W, H

    if init in {'nndsvd', 'nndsvda', 'nndsvdar'}:
        model = NMF(n_components=n_components, init=init, random_state=random_state)
        W = model._initialize_nmf(X, n_components, init=init, eps=eps, random_state=random_state)[0]
        H = model._initialize_nmf(X, n_components, init=init, eps=eps, random_state=random_state)[1]
        W[W < eps] = 0
        H[H < eps] = 0
        return W, H

    # Default initialization
    W = np.abs(rng.randn(n_samples, n_components))
    H = np.abs(rng.randn(n_components, n_features))
    W[W < eps] = 0
    H[H < eps] = 0
    return W, H

if __name__ == "__main__":
    # Example usage
    X = np.abs(np.random.randn(10, 8))
    n_components = 5
    init = 'nndsvd'
    eps = 1e-6
    random_state = 42

    W, H = _initialize_nmf(X, n_components, init=init, eps=eps, random_state=random_state)
    print("W:", W)
    print("H:", H)