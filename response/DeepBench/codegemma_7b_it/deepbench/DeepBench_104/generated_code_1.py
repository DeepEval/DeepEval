import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigh, svds

def tswspectral_layout(data, graph, dim, random_state=None, metric="euclidean",
                     metric_kwds={}, method="random", tol=1e-4, maxiter=20):
    """
    Computes the spectral embedding of a graph.

    Parameters
    ----------
    data : array_like, shape (N,)
        Input data points.
    graph : NetworkX graph
        Graph object.
    dim : int
        Dimensionality of the embedding.
    random_state : int, RandomState
        Random state for reproducibility.
    metric : str, callable
        Metric to compute distances between data points.
    metric_kwds : dict
        Keyword arguments for the distance function.
    method : str
        Method for initializing the eigenvectors. Can be 'random' or 'networkx'.
    tol : float
        Convergence tolerance for the truncated SVD.
    maxiter : int
        Maximum number of iterations for the truncated SVD.

    Returns
    -------
    array_like, shape (N, dim)
        Spectral embedding of the graph.
    """

    # Create a normalized Laplacian of the graph
    laplacian = nx.normalized_laplacian_matrix(graph)

    # Initialize eigenvectors using the specified method
    if method == "random":
        eigenvectors = np.random.normal(size=(graph.number_of_nodes(), dim))
    elif method == "networkx":
        eigenvectors = nx.eigenvector_centrality_numpy(graph)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Truncated SVD to find the eigenvectors
    for _ in range(maxiter):
        # Compute the projection matrix
        projection = eigenvectors.T @ laplacian @ eigenvectors

        # Update the eigenvectors using the projection matrix
        eigenvectors = projection @ eigenvectors

        # Check for convergence
        if np.linalg.norm(eigenvectors - projection @ eigenvectors) < tol:
            break

    return eigenvectors

if __name__ == "__main__":
    # Create sample input values
    data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data)))
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

    # Call the function and print the results
    embedding = tswspectral_layout(data, graph, dim=2)
    print(embedding)