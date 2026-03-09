import numpy as np
from sklearn.decomposition import NMF

def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    if init is None:
        init = 'random'
    
    rng = np.random.RandomState(random_state)
    
    if init == 'random':
        W = rng.rand(X.shape[0], n_components)
        H = rng.rand(n_components, X.shape[1])
    elif init == 'nndsvd':
        model = NMF(n_components=n_components, init='nndsvd', random_state=random_state)
        W, H = model.fit_transform(X), model.components_
    elif init == 'nndsvda':
        model = NMF(n_components=n_components, init='nndsvda', random_state=random_state)
        W, H = model.fit_transform(X), model.components_
    elif init == 'nndsvdar':
        model = NMF(n_components=n_components, init='nndsvdar', random_state=random_state)
        W, H = model.fit_transform(X), model.components_
    
    W[W < eps] = 0
    H[H < eps] = 0
    
    return W, H

if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [3, 3]])
    n_components = 2
    W, H = _initialize_nmf(X, n_components, init='random', eps=1e-6, random_state=42)
    print("W:\n", W)
    print("H:\n", H)