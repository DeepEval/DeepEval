import torch

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
    if degrees:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)
    
    distance = torch.tensor(distance, device=device).view(-1, 1)
    elevation = torch.tensor(elevation, device=device).view(-1, 1)
    azimuth = torch.tensor(azimuth, device=device).view(-1, 1)
    
    x = distance * torch.cos(elevation) * torch.cos(azimuth)
    y = distance * torch.cos(elevation) * torch.sin(azimuth)
    z = distance * torch.sin(elevation)
    
    return torch.cat((x, y, z), dim=1)

if __name__ == "__main__":
    distance = [10.0, 15.0]
    elevation = [30.0, 45.0]
    azimuth = [60.0, 90.0]
    
    camera_positions = camera_position_from_spherical_angles(distance, elevation, azimuth)
    print(camera_positions)