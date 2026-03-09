import torch

def unproject_points_orthographic(points_in_camera, extension):
    return torch.cat([points_in_camera, extension.repeat(1, 2)], dim=-1)

if __name__ == "__main__":
    # Create sample input values
    points_in_camera = torch.tensor([[1, 2], [3, 4], [5, 6]])
    extension = torch.tensor([[1]])

    # Call the function
    unprojected_points = unproject_points_orthographic(points_in_camera, extension)

    # Print the results
    print(unprojected_points)