import torch

def unproject_points_orthographic(points_in_camera, extension):
    points_in_camera = points_in_camera.unsqueeze(-1)  # Add a dimension to make it (..., 2, 1)
    unprojected_points = torch.cat((points_in_camera, extension), dim=-1)  # Concatenate to get shape (..., 3)
    return unprojected_points

if __name__ == "__main__":
    points_in_camera = torch.tensor([[0.5, 0.5], [0.1, -0.2], [-0.3, 0.7]])
    extension = torch.tensor([[2.0], [1.5], [1.0]])
    result = unproject_points_orthographic(points_in_camera, extension)
    print(result)