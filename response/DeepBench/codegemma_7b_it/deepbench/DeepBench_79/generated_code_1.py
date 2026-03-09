import torch

def unproject_points_orthographic(points_in_camera, extension):
    """
    Unprojects points from the canonical z=1 plane into the camera frame.

    Args:
        points_in_camera: Tensor representing the points to unproject with shape (..., 2).
        extension: Tensor representing the extension of the points to unproject with shape (..., 1).

    Returns:
        Tensor representing the unprojected points with shape (..., 3).
    """

    # Convert points to homogeneous coordinates.
    points_in_camera_homogeneous = torch.cat((points_in_camera, torch.ones_like(points_in_camera[..., 0:1])), dim=-1)

    # Unproject points using the extension.
    unprojected_points = points_in_camera_homogeneous / (extension + points_in_camera_homogeneous[..., 2:3])

    # Extract the x and y coordinates.
    unprojected_points_xy = unprojected_points[..., 0:2]

    return unprojected_points_xy

if __name__ == "__main__":
    # Sample input values.
    points_in_camera = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [1.5, -1.5]])
    extension = torch.tensor([0.2, 0.3, 0.4])

    # Call the function.
    unprojected_points = unproject_points_orthographic(points_in_camera, extension)

    # Print the results.
    print(unprojected_points)