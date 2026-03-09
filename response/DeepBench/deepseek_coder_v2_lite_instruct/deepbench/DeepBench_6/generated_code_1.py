import torch
import math

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
    # Handle broadcasting for inputs
    distance = torch.as_tensor(distance, dtype=torch.float32, device=device).reshape(-1, 1)
    elevation = torch.as_tensor(elevation, dtype=torch.float32, device=device).reshape(-1, 1)
    azimuth = torch.as_tensor(azimuth, dtype=torch.float32, device=device).reshape(-1, 1)

    # Convert degrees to radians if necessary
    if degrees:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)

    # Calculate the Cartesian coordinates
    x = distance * torch.sin(elevation) * torch.cos(azimuth)
    y = distance * torch.sin(elevation) * torch.sin(azimuth)
    z = distance * torch.cos(elevation)

    # Combine x, y, z into a single tensor
    camera_position = torch.cat((x, y, z), dim=1)

    return camera_position

if __name__ == "__main__":
    # Example usage
    distance = [10, 20, 30]
    elevation = [30, 45, 60]
    azimuth = [45, 90, 135]
    degrees = True
    device = "cpu"

    camera_pos = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees, device)
    print(camera_pos)