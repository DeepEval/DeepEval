import torch

def depth_to_3d(depth_tensor, camera_matrix, normalize_points=False):
    if not isinstance(depth_tensor, torch.Tensor) or not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Both depth_tensor and camera_matrix must be torch.Tensor instances.")
    
    if depth_tensor.ndim != 4 or camera_matrix.ndim != 3:
        raise ValueError("Incorrect dimensions for depth_tensor or camera_matrix.")
    
    B, C, H, W = depth_tensor.shape
    if C != 1:
        raise ValueError("The depth_tensor should have a channel size of 1.")
    
    if camera_matrix.shape != (B, 3, 3):
        raise ValueError("camera_matrix must have a shape of (B, 3, 3).")

    # Create a mesh grid for the pixel coordinates
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y = y.to(depth_tensor.device)
    x = x.to(depth_tensor.device)
    ones = torch.ones_like(x)

    # Stack to create homogeneous pixel coordinates
    pixel_coords = torch.stack([x, y, ones], dim=0).float()

    # Expand dimensions to match batch size
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)

    # Reshape to (B, 3, H*W) for matrix multiplication
    pixel_coords = pixel_coords.view(B, 3, H * W)

    # Inverse of camera matrix
    camera_matrix_inv = torch.inverse(camera_matrix)

    # Project pixel coordinates to camera coordinates
    cam_coords = camera_matrix_inv.bmm(pixel_coords)

    # Apply depth to get 3D points in camera coordinates
    depth_flat = depth_tensor.view(B, 1, H * W)
    points_3d = cam_coords * depth_flat

    # Reshape to (B, 3, H, W)
    points_3d = points_3d.view(B, 3, H, W)

    # Optionally normalize
    if normalize_points:
        points_3d = torch.nn.functional.normalize(points_3d, dim=1)

    return points_3d

if __name__ == "__main__":
    B, H, W = 1, 2, 2  # Batch size, height, width
    depth_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    camera_matrix = torch.tensor([[[1.0, 0.0, 0.5],
                                   [0.0, 1.0, 0.5],
                                   [0.0, 0.0, 1.0]]], dtype=torch.float32)

    points_3d = depth_to_3d(depth_tensor, camera_matrix, normalize_points=False)
    print("3D Points:\n", points_3d)