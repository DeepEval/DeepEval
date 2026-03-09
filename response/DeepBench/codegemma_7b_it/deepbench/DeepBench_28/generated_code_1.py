import torch

def depth_to_normals(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = True) -> torch.Tensor:
    """
    Computes surface normals per pixel from a depth image.

    Args:
        depth: A tensor of shape (B, 1, H, W) containing depth values.
        camera_matrix: A tensor of shape (B, 3, 3) representing the camera intrinsics.
        normalize_points: A boolean indicating whether to normalize the point cloud. This is necessary when depth values represent Euclidean distances from the camera position.

    Returns:
        A tensor with a normal vector per pixel of shape (B, 3, H, W).
    """

    if not isinstance(depth, torch.Tensor) or not isinstance(camera_matrix, torch.Tensor):
        raise TypeError("Input tensors must be of type torch.Tensor")

    if depth.shape != (1, depth.shape[2], depth.shape[3]) or camera_matrix.shape != (3, 3):
        raise ValueError("Invalid input tensor shapes")

    device = depth.device
    dtype = depth.dtype

    # Assuming the camera_matrix is in the format [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

    # Calculate the depth gradient in image space
    dx = depth / fx
    dy = depth / fy

    # Compute the surface normal in camera space
    if normalize_points:
        z = depth.squeeze(1)
        camera_origin_point = z.new_zeros((2))
        camera_origin_point[0] = cx
        camera_origin_point[1] = cy
        normal_camera_space = torch.stack([-dy, dx, -torch.ones_like(z)], dim=-1).transpose(1, 2).unsqueeze(1)
        normal_camera_space = normal_camera_space.to(device)
        camera_origin_point = camera_origin_point.to(device)
        normal_camera_space = normal_camera_space - camera_origin_point
        normal_camera_space = normal_camera_space / torch.linalg.norm(normal_camera_space, dim=-1, keepdim=True)
        return normal_camera_space.to(dtype)
    else:
        return torch.stack([-dy, dx, -torch.ones_like(depth)], dim=-1).transpose(1, 2).unsqueeze(1).to(dtype)

if __name__ == "__main__":
    # Sample input values
    depth = torch.tensor([[[5.0], [10.0], [15.0]], [[2.0], [4.0], [6.0]]], dtype=torch.float32)
    camera_matrix = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    # Call the function and print the results
    normals = depth_to_normals(depth, camera_matrix)
    print(normals)