import torch
import math

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device='cpu'):
    # Convert angles to radians if necessary
    if degrees:
        elevation = elevation * math.pi / 180
        azimuth = azimuth * math.pi / 180

    # Create a tensor to store the camera positions
    N = distance.shape[0]
    camera_positions = torch.zeros((N, 3), device=device)

    # Calculate the camera positions
    camera_positions[:, 0] = distance * torch.cos(elevation) * torch.cos(azimuth)
    camera_positions[:, 1] = distance * torch.cos(elevation) * torch.sin(azimuth)
    camera_positions[:, 2] = distance * torch.sin(elevation)

    return camera_positions

if __name__ == "__main__":
    # Create sample input values
    distance = torch.tensor([1.0, 2.0, 3.0])
    elevation = torch.tensor([30.0, 45.0, 60.0])
    azimuth = torch.tensor([45.0, 60.0, 30.0])

    # Call the function and print the results
    result_degrees = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True)
    result_radians = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False)

    print("Camera positions in degrees (torch.Tensor):")
    print(result_degrees)
    print("\nCamera positions in radians (torch.Tensor):")
    print(result_radians)