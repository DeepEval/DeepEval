import numpy as np

def depth_to_3d(depth_tensor, camera_matrix, normalize_points=False):
    """
    Generates 3D points per pixel based on depth values and camera intrinsics.
    
    Args:
        depth_tensor (numpy.ndarray): Depth values tensor with shape (B, 1, H, W).
        camera_matrix (numpy.ndarray): Camera intrinsic matrix tensor with shape (B, 3, 3).
        normalize_points (bool): If True, normalize the computed 3D points by the depth.
    
    Returns:
        numpy.ndarray: 3D point per pixel tensor with shape (B, 3, H, W).
    """
    # Check if inputs are of the correct type
    if not isinstance(depth_tensor, np.ndarray) or not isinstance(camera_matrix, np.ndarray):
        raise ValueError("Both depth_tensor and camera_matrix must be numpy.ndarray.")
    
    # Check if shapes match the expected format
    if depth_tensor.ndim != 4 or camera_matrix.ndim != 4:
        raise ValueError("depth_tensor and camera_matrix must be 4-dimensional tensors.")
    
    # Check if depth tensor's shape is (B, 1, H, W) and camera matrix's shape is (B, 3, 3)
    if depth_tensor.shape[1] != 1 or camera_matrix.shape[1] != 3 or camera_matrix.shape[2] != 3:
        raise ValueError("depth_tensor must have a depth value per pixel with shape (B, 1, H, W), and camera_matrix must have a shape (B, 3, 3).")
    
    # Generate 3D points based on depth values and camera intrinsics
    B, _, H, W = depth_tensor.shape
    three_d_points = np.zeros((B, 3, H, W))
    
    for b in range(B):
        R = camera_matrix[b, :3, :3]
        t = camera_matrix[b, :3, 3].reshape(1, 3)
        depth = depth_tensor[b]
        
        # Convert depth to homogeneous coordinates
        homogeneous_coords = np.concatenate([np.ones((1, H, W)), R, t])
        
        # Invert the camera matrix
        inverse_matrix = np.linalg.inv(camera_matrix[b])
        projected_coords = np.dot(inverse_matrix, homogeneous_coords.T).T
        
        # Convert to 3D points
        three_d_points[b] = np.dot(projected_coords[0:3], projected_coords[3:6].T).T
    
    # Optionally normalize the 3D points
    if normalize_points:
        three_d_points = three_d_points / depth_tensor
    
    return three_d_points

if __name__ == "__main__":
    # Sample input values
    B = 2
    H = 3
    W = 4
    depth_tensor = np.random.rand(B, 1, H, W)
    camera_matrix = np.random.rand(B, 3, 3)
    
    # Call the function and print the results
    three_d_points = depth_to_3d(depth_tensor, camera_matrix, normalize_points=True)
    print("3D Points:\n", three_d_points)