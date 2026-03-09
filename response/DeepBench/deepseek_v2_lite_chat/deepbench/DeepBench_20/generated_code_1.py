import numpy as np

def masked_gather(points, idx, num_points):
    if points.shape[:-1] != idx.shape[:-1]:
        raise ValueError("points and idx must have matching batch dimensions.")
    
    # Expand indices to match the shape of points
    idx_expanded = np.expand_dims(idx, axis=-1)
    
    # Replace -1 indices with 0 and set to 0.0 if they were -1
    mask = idx_expanded == -1
    points[mask] = 0.0
    
    # Gather points using the expanded indices
    gathered_points = np.take_along_axis(points, idx_expanded, axis=-1)
    
    return gathered_points

if __name__ == "__main__":
    # Sample input values
    points = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                       [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    idx = np.array([[0, -1, 2], [1, -1, 0]])
    num_points = points.shape[1]
    
    # Call the function and print the results
    masked_indices = masked_gather(points, idx, num_points)
    print("Masked indices:", masked_indices)