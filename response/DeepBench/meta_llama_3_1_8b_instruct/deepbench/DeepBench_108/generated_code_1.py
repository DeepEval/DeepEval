import numpy as np
import torch
import torchvision

def depth_to_3d(depth, camera_matrix, normalize_points):
    # Error handling for input types
    if not isinstance(depth, torch.Tensor) or not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Both depth and camera_matrix must be torch tensors.")
    if not isinstance(normalize_points, bool):
        raise TypeError("normalize_points must be a boolean.")
    
    # Error handling for input shapes
    if depth.shape[1]!= 1 or depth.shape[2]!= depth.shape[3]:
        raise ValueError("Depth tensor must have a shape of (B, 1, H, W).")
    if camera_matrix.shape[1]!= 3 or camera_matrix.shape[2]!= 3:
        raise ValueError("Camera matrix tensor must have a shape of (B, 3, 3).")
    
    # Compute 3D points
    batch_size, _, height, width = depth.shape
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    
    x_3d = depth * camera_matrix[:, 0, 0] * x / (camera_matrix[:, 2, 2] + 1e-6)
    y_3d = depth * camera_matrix[:, 1, 1] * y / (camera_matrix[:, 2, 2] + 1e-6)
    z_3d = depth * camera_matrix[:, 2, 2] / (camera_matrix[:, 2, 2] + 1e-6)
    
    if normalize_points:
        max_x = torch.max(x_3d)
        max_y = torch.max(y_3d)
        max_z = torch.max(z_3d)
        
        x_3d = x_3d / max_x
        y_3d = y_3d / max_y
        z_3d = z_3d / max_z
    
    # Stack 3D points
    points_3d = torch.stack([x_3d, y_3d, z_3d], dim=1)
    
    return points_3d

if __name__ == "__main__":
    # Create sample input values
    batch_size = 1
    height = 256
    width = 256
    
    depth = torch.randn(batch_size, 1, height, width)
    camera_matrix = torch.randn(batch_size, 3, 3)
    
    # Call the function
    points_3d = depth_to_3d(depth, camera_matrix, True)
    
    # Print the results
    print(points_3d.shape)
    print(points_3d[0, 0, 0, 0])