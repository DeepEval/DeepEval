import torch

def project_points_orthographic(points_in_camera):
    # Extract x and y coordinates from the input points
    x = points_in_camera[:, 0]
    y = points_in_camera[:, 1]
    z = points_in_camera[:, 2]
    
    # Perform orthographic projection
    u = x / (z + 1e-6)  # Avoid division by zero
    v = y / (z + 1e-6)  # Avoid division by zero
    
    # Stack the results back into a tensor
    points_in_image = torch.stack((u, v), dim=-1)
    
    return points_in_image

if __name__ == "__main__":
    # Example points in camera frame
    points_in_camera = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Project points to the canonical z=1 plane
    projected_points = project_points_orthographic(points_in_camera)
    
    # Print the results
    print("Projected points:\n", projected_points)