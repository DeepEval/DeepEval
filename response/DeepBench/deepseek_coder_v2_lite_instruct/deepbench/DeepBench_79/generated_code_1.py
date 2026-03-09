import torch

def unproject_points_orthographic(points_in_camera, extension):
    # Create a canonical z=1 plane
    canonical_points = torch.cat([points_in_camera, torch.ones_like(extension)], dim=-1)
    
    # Unproject points from z=1 plane to camera frame
    unprojected_points = canonical_points * extension
    
    return unprojected_points

if __name__ == "__main__":
    # Sample input values
    points_in_camera = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    extension = torch.tensor([[1.0], [1.0]])
    
    # Call the function
    result = unproject_points_orthographic(points_in_camera, extension)
    
    # Print the results
    print(result)