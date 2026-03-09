import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx

def pixel_to_pixel_graph(n_x, n_y, n_z=1, mask=None, return_as=sparse.coo_matrix, dtype=int):
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)
    
    # Create a boolean 3D array of the same shape as the mask to store the connections
    connections = np.zeros_like(mask, dtype=bool)
    
    # Define the directions to check for connections in the x, y, and z axes
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    # Iterate over each voxel in the mask
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                if mask[i, j, k]:  # Check if the voxel is included in the mask
                    for direction in directions:
                        ni, nj, nk = i + direction[0], j + direction[1], k + direction[2]
                        if (0 <= ni < n_x) and (0 <= nj < n_y) and (0 <= nk < n_z) and mask[ni, nj, nk]:
                            connections[i, j, k] = True
                            connections[ni, nj, nk] = True
    
    # Create a sparse matrix from the connections
    graph = return_as(connections.astype(dtype))
    
    return graph

if __name__ == "__main__":
    n_x, n_y, n_z = 5, 5, 1
    mask = np.zeros((n_x, n_y, n_z), dtype=bool)
    mask[1, 1, 0] = True
    mask[2, 2, 0] = True
    mask[3, 3, 0] = True
    
    graph = pixel_to_pixel_graph(n_x, n_y, n_z, mask)
    
    # Print the graph
    print(graph.toarray())
    
    # Visualize the graph using networkx
    G = nx.Graph()
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                if mask[i, j, k]:
                    for direction in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                        ni, nj, nk = i + direction[0], j + direction[1], k + direction[2]
                        if (0 <= ni < n_x) and (0 <= nj < n_y) and (0 <= nk < n_z) and mask[ni, nj, nk]:
                            G.add_edge((i, j, k), (ni, nj, nk))
    
    pos = {}
    for node in G.nodes:
        pos[node] = (node[0] - n_x // 2, node[1] - n_y // 2)
    
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos)
    plt.show()