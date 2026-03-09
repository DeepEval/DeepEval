import math
import torch

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
    if degrees:
        elevation = math.radians(elevation)
        azimuth = math.radians(azimuth)

    distance = torch.tensor(distance).to(device)
    elevation = torch.tensor(elevation).to(device)
    azimuth = torch.tensor(azimuth).to(device)

    x = distance * torch.cos(elevation) * torch.sin(azimuth)
    y = distance * torch.cos(elevation) * torch.cos(azimuth)
    z = distance * torch.sin(elevation)

    return torch.stack((x, y, z), dim=1)

if __name__ == "__main__":
    # Test the function with sample input values
    distance = 5.0
    elevation = 30.0
    azimuth = 45.0
    degrees = True
    device = "cpu"

    result = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees, device)
    print(result)