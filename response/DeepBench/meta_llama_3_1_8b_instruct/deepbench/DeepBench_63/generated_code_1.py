import torch

def project_points_orthographic(points_in_camera):
    # Assuming the input points are of shape (batch_size, 3) where the 3rd dimension is the depth (z)
    # We select the first two dimensions (x, y) and return them as the projected points
    projected_points = points_in_camera[:, :2] / points_in_camera[:, 2:]
    return projected_points

if __name__ == "__main__":
    # Create a batch of 3 points in the camera frame with depth
    points_in_camera = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    
    # Call the function to project the points
    projected_points = project_points_orthographic(points_in_camera)
    
    # Print the projected points
    print(projected_points)