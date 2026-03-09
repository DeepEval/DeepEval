import torch

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
    if degrees:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)
    
    x = distance * torch.sin(elevation) * torch.cos(azimuth)
    y = distance * torch.sin(elevation) * torch.sin(azimuth)
    z = distance * torch.cos(elevation)
    
    return torch.stack([x, y, z], dim=-1).to(device)

if __name__ == "__main__":
    distance = torch.tensor(5.0)
    elevation = torch.tensor(60.0)
    azimuth = torch.tensor(30.0)
    
    camera_location = camera_position_from_spherical_angles(distance, elevation, azimuth)
    
    print(camera_location)