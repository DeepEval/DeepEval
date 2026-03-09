import torch

def depth_to_3d(depth, camera_matrix, normalize_points=False):
    """
    Computes a 3D point per pixel from depth values and camera intrinsics.

    Args:
        depth (torch.Tensor): Depth values per pixel (B, 1, H, W).
        camera_matrix (torch.Tensor): Camera intrinsics (B, 3, 3).
        normalize_points (bool): Whether to normalize points by depth.

    Returns:
        torch.Tensor: 3D points per pixel (B, 3, H, W).
    """

    # Input type and shape validation
    if not isinstance(depth, torch.Tensor):
        raise TypeError("Input depth must be a torch.Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Input camera_matrix must be a torch.Tensor")
    if depth.shape[1] != 1 or depth.ndim != 4:
        raise ValueError("Input depth must have shape (B, 1, H, W)")
    if camera_matrix.shape[1] != 3 or camera_matrix.shape[2] != 3 or camera_matrix.ndim != 3:
        raise ValueError("Input camera_matrix must have shape (B, 3, 3)")

    # Unpack camera intrinsics
    fx, fy, cx, cy = camera_matrix[:, 0, 0], camera_matrix[:, 0, 1], camera_matrix[:, 0, 2], camera_matrix[:, 1, 2]

    # Reshape depth and camera intrinsics
    depth_reshaped = depth.reshape(-1, 1, depth.shape[2], depth.shape[3])
    fx_reshaped = fx.reshape(-1, 1, 1, 1)
    fy_reshaped = fy.reshape(-1, 1, 1, 1)
    cx_reshaped = cx.reshape(-1, 1, 1, 1)
    cy_reshaped = cy.reshape(-1, 1, 1, 1)

    # Calculate 3D points
    x = (depth_reshaped * (torch.arange(depth.shape[3]).to(depth) - cx_reshaped)) / fx_reshaped
    y = (depth_reshaped * (torch.arange(depth.shape[2]).to(depth) - cy_reshaped)) / fy_reshaped
    z = depth_reshaped

    # Normalize points by depth
    if normalize_points:
        x /= depth_reshaped
        y /= depth_reshaped
        z /= depth_reshaped

    # Reshape and return 3D points
    output = torch.stack([x, y, z], dim=1).reshape(-1, 3, depth.shape[2], depth.shape[3])

    return output

if __name__ == "__main__":
    # Sample input values
    depth = torch.randn(1, 1, 5, 5)
    camera_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Call the function and print the results
    points = depth_to_3d(depth, camera_matrix, normalize_points=True)
    print(points)