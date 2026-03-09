import numpy as np
from scipy.sparse import csgraph
from sklearn.manifold import SpectralEmbedding

def tswspectral_layout(data=None, graph=None, dim=2, random_state=None, metric='precomputed', metric_kwds=None, method='arpack', tol=0, maxiter=None):
    if graph is not None:
        if metric == 'precomputed':
            laplacian = csgraph.laplacian(graph, normed=True)
        else:
            distances = csgraph.shortest_path(graph, method=method, metric=metric, metric_kwds=metric_kwds)
            laplacian = csgraph.laplacian(distances, normed=True)
    elif data is not None:
        if metric == 'precomputed':
            if data.shape[0] != data.shape[1]:
                raise ValueError("Precomputed metric requires a square matrix.")
            laplacian = csgraph.laplacian(data, normed=True)
        else:
            embedding = SpectralEmbedding(n_components=dim, random_state=random_state, affinity=metric, metric_params=metric_kwds)
            data_embedding = embedding.fit_transform(data)
            return data_embedding
    else:
        raise ValueError("Either 'data' or 'graph' must be provided.")

    svd = TruncatedSVD(n_components=dim, random_state=random_state, tol=tol, max_iter=maxiter)
    embedding = svd.fit_transform(laplacian)
    return embedding

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create a sample graph (adjacency matrix)
    graph = np.array([[0, 1, 0, 0],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])

    # Call the function with the sample graph
    embedding = tswspectral_layout(graph=graph)

    # Print the results
    print(embedding)