import torch

def depth_to_normals(depth, camera_matrix, normalize_points=True):
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a Tensor")
    if depth.shape != (depth.shape[0], 1, depth.shape[2], depth.shape[3]):
        raise ValueError("depth must have shape (B, 1, H, W)")
    if camera_matrix.shape != (camera_matrix.shape[0], 3, 3):
        raise ValueError("camera_matrix must have shape (B, 3, 3)")

    B, _, H, W = depth.shape
    device = depth.device

    # Create grid of (H, W, 2) in the range [-1, 1]
    i, j = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                          torch.arange(W, dtype=torch.float32, device=device))
    grid = torch.stack((j, i), dim=-1).view(1, H, W, 2)  # (1, H, W, 2)

    # Normalize the grid to the range [0, 1]
    grid = (grid + 1) / 2  # (1, H, W, 2)

    # Unproject the grid to 3D points
    points = torch.nn.functional.unpad(grid, (W % 2, H % 2)).permute(0, 3, 1, 2)  # (B, 2, H, W)
    points = points.unsqueeze(1)  # (B, 1, 2, H, W)
    points = points * depth  # (B, 1, 2, H, W) * (B, 1, 1, H, W) -> (B, 1, 2, H, W)

    # Transform points to world coordinates
    points = torch.cat((points, depth.unsqueeze(1)), dim=1)  # (B, 2, H, W) -> (B, 3, H, W)
    points = points.view(B, 3, -1).permute(0, 2, 1)  # (B, 3, H, W)
    points = torch.bmm(camera_matrix.view(B, 3, 3), points).permute(0, 2, 1)  # (B, 3, 3) @ (B, 3, H*W) -> (B, 3, H*W)
    points = points.view(B, 3, H, W)  # (B, 3, H, W)

    # Compute gradients to find surface normals
    dz_dx = torch.abs(points[:, :, 1:, :] - points[:, :, :-1, :])
    dz_dy = torch.abs(points[:, :, :, 1:] - points[:, :, :, :-1])

    normals = torch.cross(torch.stack((dz_dx, torch.zeros_like(dz_dx), -dz_dy), dim=1),
                          torch.stack((-dz_dx, torch.zeros_like(dz_dx), dz_dy), dim=1), dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)

    if normalize_points:
        points = points / depth.unsqueeze(1)

    return normals

if __name__ == "__main__":
    # Create sample input values
    depth = torch.rand(1, 1, 4, 4)
    camera_matrix = torch.rand(1, 3, 3)

    # Call the function
    normals = depth_to_normals(depth, camera_matrix)

    # Print the results
    print(normals)