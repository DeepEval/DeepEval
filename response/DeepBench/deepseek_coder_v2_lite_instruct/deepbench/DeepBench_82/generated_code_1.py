import numpy as np
from scipy.sparse import coo_matrix

def create_graph(n_x, n_y, n_z=1, mask=None, return_as=coo_matrix, dtype=int):
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)
    
    # Create a 3D grid of indices
    x, y, z = np.meshgrid(np.arange(n_x), np.arange(n_y), np.arange(n_z), indexing='ij')
    indices = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    
    # Create a list of edges (i, j) where i and j are connected
    edges = []
    for i in range(indices.shape[0]):
        for j in range(i+1, indices.shape[0]):
            if np.sum(np.abs(indices[i] - indices[j])) == 1:
                edges.append((i, j))
    
    # Create the adjacency matrix in the specified format
    data = np.ones(len(edges), dtype=dtype)
    row = np.array([e[0] for e in edges]).astype(int)
    col = np.array([e[1] for e in edges]).astype(int)
    graph = return_as((row, col, data), shape=(indices.shape[0], indices.shape[0]))
    
    return graph

if __name__ == "__main__":
    n_x, n_y = 3, 3
    graph = create_graph(n_x, n_y)
    print(graph.toarray())