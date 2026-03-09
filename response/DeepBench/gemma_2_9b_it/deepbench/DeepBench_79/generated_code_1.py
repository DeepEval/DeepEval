import torch

def unproject_points_orthographic(points_in_camera, extension):
  points_in_camera = torch.cat([points_in_camera, extension], dim=-1)
  return points_in_camera

if __name__ == "__main__":
  points_in_camera = torch.randn(10, 2)
  extension = torch.randn(10, 1)
  unprojected_points = unproject_points_orthographic(points_in_camera, extension)
  print(unprojected_points)