import torch

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
    if degrees:
        elevation = torch.deg2rad(torch.tensor(elevation, device=device))
        azimuth = torch.deg2rad(torch.tensor(azimuth, device=device))
    else:
        elevation = torch.tensor(elevation, device=device)
        azimuth = torch.tensor(azimuth, device=device)

    distance = torch.tensor(distance, device=device)
    
    # Ensure all inputs are broadcasted to the same shape
    distance = distance.view(-1, 1)
    elevation = elevation.view(-1, 1)
    azimuth = azimuth.view(-1, 1)
    
    # Spherical to Cartesian conversion
    x = distance * torch.cos(elevation) * torch.cos(azimuth)
    y = distance * torch.cos(elevation) * torch.sin(azimuth)
    z = distance * torch.sin(elevation)
    
    camera_position = torch.cat([x, y, z], dim=1)
    
    return camera_position

if __name__ == "__main__":
    distance = 10.0
    elevation = 45.0
    azimuth = 30.0
    camera_pos = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu")
    print("Camera Position (Single Input):", camera_pos)

    # Example with multiple inputs
    distances = [10.0, 15.0]
    elevations = [45.0, 60.0]
    azimuths = [30.0, 90.0]
    camera_pos_multiple = camera_position_from_spherical_angles(distances, elevations, azimuths, degrees=True, device="cpu")
    print("Camera Positions (Multiple Inputs):", camera_pos_multiple)