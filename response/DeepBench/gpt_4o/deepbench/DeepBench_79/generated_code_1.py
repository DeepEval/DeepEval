import torch

def unproject_points_orthographic(points_in_camera, extension):
    # Concatenate the input points with the extension along the last dimension
    unprojected_points = torch.cat((points_in_camera, extension), dim=-1)
    return unprojected_points

if __name__ == "__main__":
    # Create sample input values
    points_in_camera = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    extension = torch.tensor([[5.0], [6.0]])

    # Call the function
    unprojected_points = unproject_points_orthographic(points_in_camera, extension)

    # Print the results
    print("Unprojected Points:")
    print(unprojected_points)