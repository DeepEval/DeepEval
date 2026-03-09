import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigs

def tswspectral_layout(data, graph, dim, random_state, metric, metric_kwds, method, tol, maxiter):
    # Compute the normalized Laplacian
    if metric == 'euclidean':
        dist = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    elif metric == 'precomputed':
        dist = graph
    else:
        raise ValueError("Unsupported metric")

    deg = np.sum(graph, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0
    deg_inv_sqrt = np.diag(deg_inv_sqrt)

    laplacian = np.eye(len(deg)) - deg_inv_sqrt @ graph @ deg_inv_sqrt

    # Compute the top-k eigenvectors of the Laplacian matrix
    k = min(dim, len(deg))
    evals, evecs = randomized_svd(laplacian, k=k, n_iter=maxiter, random_state=random_state)
    evals, evecs = evals[:k], evecs[:, :k]

    # Re-orthogonalize the eigenvectors to ensure they are orthonormal
    for i in range(evecs.shape[1]):
        for j in range(i):
            evecs[:, i] -= np.dot(evecs[:, i], evecs[:, j]) * evecs[:, j]

    # Normalize the eigenvectors
    evecs = evecs / np.linalg.norm(evecs, axis=0)

    return evecs

if __name__ == "__main__":
    # Create sample input values
    np.random.seed(0)
    data = np.random.rand(10, 2)
    graph = np.exp(-np.linalg.norm(data[:, np.newaxis] - data, axis=2))
    dim = 2
    random_state = 0
    metric = 'euclidean'
    metric_kwds = {}
    method = 'tsw'
    tol = 1e-6
    maxiter = 100

    # Call the function and print the results
    embedding = tswspectral_layout(data, graph, dim, random_state, metric, metric_kwds, method, tol, maxiter)
    print(embedding)