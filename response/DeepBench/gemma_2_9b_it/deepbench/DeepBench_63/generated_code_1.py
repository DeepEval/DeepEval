import torch

def project_points_orthographic(points_in_camera):
  z = points_in_camera[..., 2]
  return points_in_camera[..., :2] * (1 / z[..., None])

if __name__ == "__main__":
  points_in_camera = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  projected_points = project_points_orthographic(points_in_camera)
  print(projected_points)