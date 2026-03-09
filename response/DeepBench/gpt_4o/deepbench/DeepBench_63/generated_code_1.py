import torch

def project_points_orthographic(points_in_camera):
    # Extract x and y coordinates, discard z
    projected_points = points_in_camera[:, :2]
    return projected_points

if __name__ == "__main__":
    # Create a sample input tensor with shape (n, 3), where n is the number of points
    sample_points = torch.tensor([[2.0, 3.0, 5.0], [4.0, 6.0, 8.0], [7.0, 8.0, 9.0]])

    # Call the function
    projected_points = project_points_orthographic(sample_points)

    # Print the results
    print("Projected Points:\n", projected_points)