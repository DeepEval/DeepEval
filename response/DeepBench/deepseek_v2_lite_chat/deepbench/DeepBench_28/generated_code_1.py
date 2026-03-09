import torch
import torch.nn as nn
import numpy as np

def depth_to_normals(depth, camera_matrix, normalize_points):
    """
    Computes surface normals per pixel from a depth image.
    
    Args:
    - depth: A tensor of shape (B, 1, H, W) containing depth values.
    - camera_matrix: A tensor of shape (B, 3, 3) representing the camera intrinsics.
    - normalize_points: A boolean indicating whether to normalize the point cloud.
    
    Returns:
    - A tensor with a normal vector per pixel of shape (B, 3, H, W).
    """

    if not isinstance(depth, torch.Tensor):
        raise TypeError("Depth must be a torch Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Camera matrix must be a torch Tensor")
    
    B, _, H, W = depth.shape
    
    # Create a point cloud with shape (B, 3, N) where N is the number of points
    point_cloud = torch.cat([depth.reshape(B, 1, H*W), 
                             torch.zeros(B, 2, H*W).to(depth.device),
                             torch.ones(B, 1, H*W).to(depth.device)], dim=2)
    
    # Transform the point cloud from camera coordinates to world coordinates
    world_points = torch._mul(camera_matrix.inverse(), point_cloud)
    
    # Compute the surface normals
    normals = world_points[..., 2:]*0.  # Initialize normals to zero
    for i in range(3):
        normals += world_points[..., i:i+3]
    normals = normals[..., 0:2] / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
    
    # Normalize the normals if required
    if normalize_points:
        # Compute the maximum and minimum depth values
        max_depth = depth.max()
        min_depth = depth.min()
        # Normalize the normals to the range [0, 1]
        normalized_normals = (normals - min_depth) / (max_depth - min_depth)
        return normalized_normals.reshape(B, 3, H, W)
    else:
        return normals.reshape(B, 3, H, W)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample input values
    B, _, H, W = 10, 1, 200, 200
    depth = torch.from_numpy(np.random.rand(B, 1, H, W)*100).float()
    camera_matrix = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    normalize_points = True

    # Call the function and print the results
    normals = depth_to_normals(depth, camera_matrix, normalize_points)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(depth.numpy().squeeze(), origin='lower')
    plt.title('Depth Image')
    plt.subplot(2, 2, 2)
    plt.imshow(camera_matrix.numpy().squeeze(), origin='lower')
    plt.title('Camera Matrix')
    plt.subplot(2, 2, 3)
    plt.imshow(normals[0].numpy().squeeze(), origin='lower')
    plt.title('Normalized Normal Vectors (if normalize_points=True)')
    plt.subplot(2, 2, 4)
    plt.imshow(normals[1].numpy().squeeze(), origin='lower')
    plt.title('Normalized Normal Vectors (if normalize_points=False)')
    plt.show()