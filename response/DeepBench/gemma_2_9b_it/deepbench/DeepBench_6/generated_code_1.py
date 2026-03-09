import torch
import torch.nn.functional as F

def camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=True, device="cpu"):
  if degrees:
    elevation = torch.tensor(elevation, device=device) * torch.pi / 180
    azimuth = torch.tensor(azimuth, device=device) * torch.pi / 180
  else:
    elevation = torch.tensor(elevation, device=device)
    azimuth = torch.tensor(azimuth, device=device)
  
  elevation = elevation.unsqueeze(1)
  azimuth = azimuth.unsqueeze(1)

  x = distance * torch.cos(elevation) * torch.sin(azimuth)
  y = distance * torch.sin(elevation)
  z = distance * torch.cos(elevation) * torch.cos(azimuth)
  return torch.cat((x, y, z), dim=1)

if __name__ == "__main__":
  distance = 5.0
  elevation = 30.0
  azimuth = 45.0
  degrees = True
  camera_location = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees, device="cpu")
  print(camera_location)