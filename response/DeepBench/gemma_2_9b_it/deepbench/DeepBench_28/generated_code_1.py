import torch
import numpy as np

def depth_to_normals(depth, camera_matrix, normalize_points=False):
  if not isinstance(depth, torch.Tensor):
    raise TypeError("Depth must be a Tensor.")
  if not isinstance(camera_matrix, torch.Tensor):
    raise TypeError("Camera matrix must be a Tensor.")
  if depth.shape != (torch.Size([None, 1, None, None])):
    raise ValueError("Depth must have shape (B, 1, H, W).")
  if camera_matrix.shape != (torch.Size([None, 3, 3])):
    raise ValueError("Camera matrix must have shape (B, 3, 3).")
  
  
  h, w = depth.shape[2], depth.shape[3]
  i, j = torch.meshgrid(torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32), indexing='xy')
  i = i.unsqueeze(0).unsqueeze(0)
  j = j.unsqueeze(0).unsqueeze(0)
  
  depth_x = (i - w * 0.5) * depth / w
  depth_y = (j - h * 0.5) * depth / h

  x_vec = torch.autograd.grad(depth_x, depth)[0]
  y_vec = torch.autograd.grad(depth_y, depth)[0]
  
  normals = torch.stack((x_vec, y_vec, -torch.ones_like(depth_x)), dim=1)
  
  if normalize_points:
    normals = normals / torch.norm(normals, dim=1, keepdim=True) 
  
  return normals.squeeze()

if __name__ == "__main__":
  depth = torch.randn(1, 1, 128, 128)
  camera_matrix = torch.randn(1, 3, 3)
  
  normals = depth_to_normals(depth, camera_matrix)
  print(normals.shape)