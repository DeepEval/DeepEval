import numpy as np
from scipy import sparse

def pixel_to_pixel_graph(n_x, n_y, n_z=1, mask=None, return_as=sparse.coo_matrix, dtype=int):
    if mask is None:
        mask = np.ones((n_x, n_y, n_z), dtype=bool)
    
    coords = np.argwhere(mask)
    n_voxels = len(coords)
    
    row = []
    col = []
    data = []
    
    for i in range(n_voxels):
        x, y, z = coords[i]
        
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            if 0 <= nx < n_x and 0 <= ny < n_y and 0 <= nz < n_z and mask[nx, ny, nz]:
                row.append(i)
                col.append(np.argwhere((coords == [nx, ny, nz]).all(axis=1))[0][0])
                data.append(1)
    
    graph = return_as((data, (row, col)), shape=(n_voxels, n_voxels), dtype=dtype)
    
    return graph

if __name__ == "__main__":
    n_x, n_y, n_z = 3, 3, 1
    mask = np.array([[True, True, False],
                     [True, True, True],
                     [False, True, True]], dtype=bool)
    
    graph = pixel_to_pixel_graph(n_x, n_y, n_z, mask)
    print(graph.toarray())  # If using sparse matrix, convert to dense for printing