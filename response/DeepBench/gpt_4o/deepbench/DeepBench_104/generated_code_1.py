import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

def tswspectral_layout(data, graph, dim=2, random_state=None, metric='euclidean', metric_kwds=None, method='truncated_svd', tol=1e-5, maxiter=300):
    random_state = check_random_state(random_state)
    
    if metric_kwds is None:
        metric_kwds = {}

    if method != 'truncated_svd':
        raise ValueError(f"Only 'truncated_svd' method is supported, got {method}.")

    n_nodes = graph.shape[0]
    
    if isinstance(graph, np.ndarray):
        adjacency = graph
    elif sp.issparse(graph):
        adjacency = graph.toarray()
    else:
        raise ValueError("The graph should be a numpy array or a sparse matrix.")

    degree_matrix = np.diag(adjacency.sum(axis=1))
    laplacian = degree_matrix - adjacency

    d = np.sum(adjacency, axis=1)
    d[d == 0] = 1
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    normalized_laplacian = np.dot(np.dot(d_inv_sqrt, laplacian), d_inv_sqrt)

    u, s, vt = svds(normalized_laplacian, k=dim, tol=tol, maxiter=maxiter, which='SM', return_singular_vectors='u')

    return u

if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.neighbors import kneighbors_graph

    # Example data
    data, _ = make_moons(n_samples=100, noise=0.1, random_state=0)
    graph = kneighbors_graph(data, n_neighbors=10, mode='connectivity', include_self=True)

    # Compute spectral layout
    embedding = tswspectral_layout(data, graph, dim=2, random_state=42)

    print("Spectral Embedding:\n", embedding)