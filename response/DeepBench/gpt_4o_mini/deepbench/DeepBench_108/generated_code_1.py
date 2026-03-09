import torch

def depth_to_3d(depth, camera_matrix, normalize_points=False):
    if not isinstance(depth, torch.Tensor) or not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Both depth and camera_matrix must be PyTorch tensors.")
    
    if depth.ndim != 4 or camera_matrix.ndim != 3:
        raise ValueError("Depth tensor must have shape (B, 1, H, W) and camera_matrix must have shape (B, 3, 3).")
    
    B, _, H, W = depth.shape
    if camera_matrix.shape[0] != B or camera_matrix.shape[1:] != (3, 3):
        raise ValueError("Camera matrix must have shape (B, 3, 3) where B matches depth batch size.")
    
    # Generate pixel grid
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(depth.device)  # Shape (1, H, W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(depth.device)  # Shape (1, H, W)

    # Convert pixel coordinates to normalized camera coordinates
    fx = camera_matrix[:, 0, 0].view(B, 1, 1)  # Focal length x
    fy = camera_matrix[:, 1, 1].view(B, 1, 1)  # Focal length y
    cx = camera_matrix[:, 0, 2].view(B, 1, 1)  # Principal point x
    cy = camera_matrix[:, 1, 2].view(B, 1, 1)  # Principal point y

    Z = depth  # Depth value per pixel
    X = (grid_x - cx) * Z / fx
    Y = (grid_y - cy) * Z / fy

    points_3d = torch.cat((X, Y, Z), dim=1)  # Shape (B, 3, H, W)

    if normalize_points:
        norm = torch.norm(points_3d, dim=1, keepdim=True)
        points_3d = points_3d / (norm + 1e-10)  # Avoid division by zero

    return points_3d

if __name__ == "__main__":
    B, H, W = 2, 4, 4
    depth = torch.rand(B, 1, H, W) * 10  # Random depth values
    camera_matrix = torch.tensor([[[800, 0, 320], [0, 800, 240], [0, 0, 1]],
                                   [[800, 0, 320], [0, 800, 240], [0, 0, 1]]], dtype=torch.float32)
    
    points_3d = depth_to_3d(depth, camera_matrix, normalize_points=False)
    print("3D Points shape:", points_3d.shape)
    print("3D Points:", points_3d)