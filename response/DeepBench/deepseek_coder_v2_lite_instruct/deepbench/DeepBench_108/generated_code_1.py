import torch

def depth_to_3d(depth, camera_matrix, normalize=False):
    batch_size, _, height, width = depth.shape
    device = depth.device

    # Check if input shapes are correct
    if camera_matrix.shape != (batch_size, 3, 3):
        raise ValueError("Camera matrix shape must be (B, 3, 3)")

    # Create grid of pixel coordinates
    x_grid, y_grid = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device))
    x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)

    # Normalize pixel coordinates to [-1, 1]
    x_grid = (x_grid / (width - 1) * 2) - 1
    y_grid = (y_grid / (height - 1) * 2) - 1

    # Convert to homogeneous coordinates
    ones = torch.ones_like(x_grid)
    pixel_coords = torch.stack([x_grid, y_grid, ones], dim=1)

    # Reshape camera matrix to (B, 3, 3) and expand to match depth shape
    camera_matrix = camera_matrix.unsqueeze(2).unsqueeze(3).expand(batch_size, 3, height, width)

    # Compute 3D points
    points_3d = torch.matmul(torch.inverse(camera_matrix), pixel_coords.unsqueeze(1))
    points_3d = points_3d * depth.unsqueeze(1)

    # Normalize if specified
    if normalize:
        points_3d = points_3d / points_3d[:, 2:3, :, :]

    return points_3d.squeeze(1)

if __name__ == "__main__":
    # Create sample input values
    batch_size = 2
    height = 4
    width = 4
    depth = torch.rand(batch_size, 1, height, width)
    camera_matrix = torch.rand(batch_size, 3, 3)

    # Call the function and print the results
    points_3d = depth_to_3d(depth, camera_matrix, normalize=True)
    print(points_3d)