import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import norm

def tswspectral_layout(data, graph, dim, random_state=None, metric='normalized_laplacian', metric_kwds={}, method='truncated_svd', tol=1e-4, maxiter=100):
    """
    Compute the spectral embedding of a graph using a truncated SVD-based approach.

    Parameters:
    - data: the data matrix of shape (n_samples, n_features)
    - graph: the graph Laplacian of shape (n_samples, n_samples)
    - dim: the dimensionality of the output embedding
    - random_state: the random state to use for random initialization
    - metric: the metric to use for computing the graph Laplacian
    - metric_kwds: additional keyword arguments to pass to the metric function
    - method: the method to use for computing the eigenvectors
    - tol: the tolerance for convergence
    - maxiter: the maximum number of iterations

    Returns:
    - the spectral embedding of the graph
    """
    # Compute the graph Laplacian
    if metric == 'normalized_laplacian':
        # Compute the normalized Laplacian
        lap = np.eye(data.shape[0]) - np.dot(data, data.T)
        lap = lap / np.sqrt(np.sum(lap ** 2))
    elif metric == 'graph_laplacian':
        # Compute the graph Laplacian
        lap = -np.eye(data.shape[0]) + np.dot(data, data.T)
    else:
        raise ValueError(f'Unknown metric {metric}')

    # Compute the eigenvectors
    if method == 'truncated_svd':
        # Compute the truncated SVD of the graph Laplacian
        u, s, vh = svds(lap, k=dim, tol=tol, maxiter=maxiter)
        # Compute the eigenvectors
        vectors = np.dot(u, vh)
    else:
        raise ValueError(f'Unknown method {method}')

    # Normalize the eigenvectors
    vectors = vectors / np.sqrt(np.sum(vectors ** 2, axis=0))

    return vectors

if __name__ == "__main__":
    # Sample data
    data = np.array([[1, 2], [3, 4]])

    # Compute the graph Laplacian
    graph = np.array([[0, 1], [2, 0]])

    # Compute the spectral embedding
    embedding = tswspectral_layout(data, graph, dim=2)

    # Print the embedding
    print(embedding)