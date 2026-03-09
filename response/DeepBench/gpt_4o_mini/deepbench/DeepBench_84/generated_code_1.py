import numpy as np
from sklearn.decomposition import NMF

def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    if init == 'random':
        W = np.random.rand(X.shape[0], n_components)
        H = np.random.rand(n_components, X.shape[1])
    
    elif init in ['nndsvd', 'nndsvda', 'nndsvdar']:
        model = NMF(n_components=n_components, init=init, random_state=random_state)
        W = model.fit_transform(X)
        H = model.components_

    else:
        raise ValueError("Invalid initialization method. Choose from 'random', 'nndsvd', 'nndsvda', or 'nndsvdar'.")

    W[W < eps] = 0
    H[H < eps] = 0

    return W, H

if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n_components = 2
    W, H = _initialize_nmf(X, n_components, init='random', random_state=42)
    print("W:", W)
    print("H:", H)