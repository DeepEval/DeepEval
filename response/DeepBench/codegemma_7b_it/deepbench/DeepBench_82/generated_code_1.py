import numpy as np
from scipy import sparse

def pixel_to_pixel_connections(n_x, n_y, n_z=1, mask=None, return_as=sparse.coo_matrix, dtype=int):
    """
    Write a python function that return Graph of the pixel-to-pixel connections. Edges exist if 2 voxels are connected.

    Parameters ----------
    n_x : int
        Dimension in x axis.
    n_y : int
        Dimension in y axis.
    n_z : int, default=1
        Dimension in z axis.
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the pixels.
    return_as : np.ndarray or a sparse matrix class, \ default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int.

    Returns -------
    graph : np.ndarray or a sparse matrix class
        The computed adjacency matrix.
    """
    # Create a meshgrid of pixel indices
    x, y, z = np.mgrid[:n_x, :n_y, :n_z]

    # Create a mask if not provided
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)

    # Create a list of connected pixel pairs
    connections = []
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                if mask[i, j, k]:
                    # Check connections in x, y, and z directions
                    for dx in [-1, 1]:
                        if i + dx >= 0 and i + dx < n_x and mask[i + dx, j, k]:
                            connections.append((i, j, k, i + dx, j, k))
                    for dy in [-1, 1]:
                        if j + dy >= 0 and j + dy < n_y and mask[i, j + dy, k]:
                            connections.append((i, j, k, i, j + dy, k))
                    for dz in [-1, 1]:
                        if k + dz >= 0 and k + dz < n_z and mask[i, j, k + dz]:
                            connections.append((i, j, k, i, j, k + dz))

    # Create an adjacency matrix using the connections
    if return_as == sparse.coo_matrix:
        graph = sparse.coo_matrix((np.ones(len(connections)), (np.array([c[0] for c in connections]), np.array([c[1] for c in connections]))),
                             (np.array([c[2] for c in connections]), np.array([c[3] for c in connections])),
                             (n_x, n_y, n_z), dtype=dtype)
    elif return_as == np.ndarray:
        graph = np.zeros((n_x * n_y * n_z, n_x * n_y * n_z), dtype=dtype)
        for i, j in enumerate(connections):
            graph[j[0] * n_y * n_z + j[1] * n_z + j[2], j[3] * n_y * n_z + j[4] * n_z + j[5]] = 1
    else:
        raise ValueError("Invalid return_as argument.")

    return graph

if __name__ == "__main__":
    # Example usage
    n_x = 5
    n_y = 5
    n_z = 5
    mask = np.zeros((n_x, n_y, n_z), dtype=bool)
    mask[2:4, 2:4, 2:4] = True
    graph = pixel_to_pixel_connections(n_x, n_y, n_z, mask)
    print(graph)