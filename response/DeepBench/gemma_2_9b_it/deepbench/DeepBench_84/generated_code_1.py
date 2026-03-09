import numpy as np
from sklearn.decomposition import NMF

def _initialize_nmf(X, n_components, init='random', eps=1e-6, random_state=None):
    if init == 'random':
        W = np.random.rand(X.shape[0], n_components)
        H = np.random.rand(n_components, X.shape[1])
    elif init == 'nndsvd':
        nmf = NMF(n_components=n_components, init='nndsvd', random_state=random_state)
        W, H = nmf.fit_transform(X)
    elif init == 'nndsvda':
        nmf = NMF(n_components=n_components, init='nndsvda', random_state=random_state)
        W, H = nmf.fit_transform(X)
    elif init == 'nndsvdar':
        nmf = NMF(n_components=n_components, init='nndsvdar', random_state=random_state)
        W, H = nmf.fit_transform(X)
    else:
        raise ValueError(f"Invalid initialization method: {init}. Choose from 'random', 'nndsvd', 'nndsvda', 'nndsvdar'.")

    W = np.clip(W, a_min=0, a_max=None)
    H = np.clip(H, a_min=0, a_max=None)

    return W, H

if __name__ == "__main__":
    X = np.random.rand(10, 5)
    W, H = _initialize_nmf(X, n_components=3, init='nndsvd')
    print("W:\n", W)
    print("H:\n", H)