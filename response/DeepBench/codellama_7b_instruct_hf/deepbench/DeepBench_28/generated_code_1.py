import torch
import torch.nn.functional as F

def depth_to_normals(depth, camera_matrix, normalize_points=False):
    """
    Computes the surface normals per pixel from a depth image.

    Args:
        depth (torch.Tensor): A tensor of shape (B, 1, H, W) containing depth values.
        camera_matrix (torch.Tensor): A tensor of shape (B, 3, 3) representing the camera intrinsics.
        normalize_points (bool, optional): Whether to normalize the point cloud. Defaults to False.

    Returns:
        torch.Tensor: A tensor with a normal vector per pixel of shape (B, 3, H, W).

    Raises:
        TypeError: If `depth` or `camera_matrix` is not a Tensor.
        ValueError: If `depth` does not have the shape (B, 1, H, W) or `camera_matrix` does not have the shape (B, 3, 3).
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a Tensor")
    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("camera_matrix must be a Tensor")
    if depth.shape != (B, 1, H, W):
        raise ValueError("depth must have the shape (B, 1, H, W)")
    if camera_matrix.shape != (B, 3, 3):
        raise ValueError("camera_matrix must have the shape (B, 3, 3)")

    # Compute the point cloud from the depth image
    points = depth_to_points(depth, camera_matrix)

    # Compute the surface normals
    normals = F.normalize(points)

    # Normalize the point cloud if requested
    if normalize_points:
        normals = F.normalize(normals)

    return normals

# Test the function
if __name__ == "__main__":
    # Create sample input values
    batch_size = 2
    height = 640
    width = 480
    depth = torch.randn(batch_size, 1, height, width)
    camera_matrix = torch.randn(batch_size, 3, 3)

    # Call the function and print the results
    normals = depth_to_normals(depth, camera_matrix)
    print(normals.shape)