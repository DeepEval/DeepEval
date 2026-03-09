import numpy as np
import scipy.sparse as sp

def function_name(n_x, n_y, n_z, mask=None, return_as=sp.coo_matrix, dtype=np.int64):
    # Create a random binary mask
    if mask is None:
        mask = np.random.choice([True, False], (n_x, n_y, n_z), p=[0.5, 0.5])
    else:
        mask = mask.astype(bool)
    
    # Create a 3D array with the mask values
    values = np.ones(mask.size, dtype=bool)
    
    # Create a boolean mask for the masked voxels
    mask_values = np.where(mask)
    
    # Create the adjacency matrix using the given return_as class and dtype
    graph = return_as((mask_values[0], mask_values[1], mask_values[1]), (mask_values[2],), dtype)
    
    return graph

if __name__ == "__main__":
    # Sample input values
    n_x = 10
    n_y = 10
    n_z = 10
    mask = None
    
    # Call the function and print the results
    graph = function_name(n_x, n_y, n_z, mask)
    print(graph.toarray())

    # Example to verify the output
    # Assuming the graph is a binary matrix (0s and 1s)
    expected_graph = np.random.randint(0, 2, (n_x, n_y, n_z))
    np.testing.assert_array_equal(graph.toarray(), expected_graph)