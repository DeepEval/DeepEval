import torch
import torch.nn.functional as F

def depth_to_normals(depth, camera_matrix, normalize_points):
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a Tensor")
    
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have the shape (B, 1, H, W)")
    if camera_matrix.ndim != 3 or camera_matrix.shape[1:] != (3, 3):
        raise ValueError("camera_matrix must have the shape (B, 3, 3)")

    B, _, H, W = depth.shape

    # Create grid of pixel coordinates
    x = torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, 1, 1, W).expand(B, 1, H, W)
    y = torch.arange(H, dtype=depth.dtype, device=depth.device).view(1, 1, H, 1).expand(B, 1, H, W)

    # Get depth values
    z = depth.squeeze(1)  # Shape (B, H, W)

    # Convert pixel coordinates to camera coordinates
    fx = camera_matrix[:, 0, 0].view(B, 1, 1, 1)
    fy = camera_matrix[:, 1, 1].view(B, 1, 1, 1)
    cx = camera_matrix[:, 0, 2].view(B, 1, 1, 1)
    cy = camera_matrix[:, 1, 2].view(B, 1, 1, 1)

    # Compute camera coordinates
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    # Stack to form the point cloud
    points = torch.stack((X, Y, Z), dim=1)  # Shape (B, 3, H, W)

    # Compute normals using gradients
    dX = F.pad(points[:, 0], (0, 1, 0, 0))[:, :-1] - F.pad(points[:, 0], (1, 0, 0, 0))[:, 1:]
    dY = F.pad(points[:, 1], (0, 0, 0, 1))[:, :, :-1] - F.pad(points[:, 1], (0, 0, 1, 0))[:, :, 1:]

    normals = torch.cross(dX, dY, dim=1)

    if normalize_points:
        normals = F.normalize(normals, p=2, dim=1)

    return normals

if __name__ == "__main__":
    B, H, W = 2, 4, 4
    depth = torch.tensor([[[[1.0, 1.0, 1.0, 1.0],
                            [[1.0, 1.0, 1.0, 1.0],
                            [[1.0, 1.0, 1.0, 1.0],
                            [[1.0, 1.0, 1.0, 1.0]]]],
                           [[1.0, 1.0, 1.0, 1.0],
                            [[1.0, 1.0, 1.0, 1.0],
                            [[1.0, 1.0, 1.0, 1.0],
                            [[1.0, 1.0, 1.0, 1.0]]]]]]]).float()
    
    camera_matrix = torch.tensor([[[500.0, 0.0, 320.0],
                                    [0.0, 500.0, 240.0],
                                    [0.0, 0.0, 1.0]],
                                   [[500.0, 0.0, 320.0],
                                    [0.0, 500.0, 240.0],
                                    [0.0, 0.0, 1.0]]]).float()
    
    normals = depth_to_normals(depth, camera_matrix, normalize_points=True)
    print(normals)