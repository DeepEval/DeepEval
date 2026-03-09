import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def create_pixel_graph(n_x, n_y, n_z=1, mask=None, return_as=coo_matrix, dtype=int):
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)

    indices = np.argwhere(mask)
    num_pixels = indices.shape[0]

    pixel_dict = {tuple(idx): i for i, idx in enumerate(indices)}
    
    rows = []
    cols = []

    for idx in indices:
        x, y, z = idx
        current_pixel = pixel_dict[(x, y, z)]

        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            neighbor = (x + dx, y + dy, z + dz)
            if 0 <= neighbor[0] < n_x and 0 <= neighbor[1] < n_y and 0 <= neighbor[2] < n_z:
                if mask[neighbor]:
                    neighbor_pixel = pixel_dict[neighbor]
                    rows.append(current_pixel)
                    cols.append(neighbor_pixel)

    data = np.ones(len(rows), dtype=dtype)
    graph = return_as((data, (rows, cols)), shape=(num_pixels, num_pixels))

    return graph

if __name__ == "__main__":
    n_x, n_y, n_z = 3, 3, 1
    mask = np.array([[[True, True, False], [True, True, True], [False, True, True]]], dtype=bool).reshape((n_x, n_y, n_z))
    graph = create_pixel_graph(n_x, n_y, n_z, mask, return_as=csr_matrix)

    print("Graph adjacency matrix (non-zero elements):")
    print(graph)