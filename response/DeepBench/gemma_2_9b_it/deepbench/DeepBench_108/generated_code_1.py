import torch

def depth_to_3d(depth, camera_matrix, normalize=True):
  if not isinstance(depth, torch.Tensor):
    raise TypeError("Depth tensor must be of type torch.Tensor")
  if depth.shape[0] != camera_matrix.shape[0]:
    raise ValueError("Batch size of depth and camera matrix must match")
  if depth.shape != (None, 1, None, None):
    raise ValueError("Depth tensor must have shape (B, 1, H, W)")
  if camera_matrix.shape != (None, 3, 3):
    raise ValueError("Camera matrix must have shape (B, 3, 3)")

  B, _, H, W = depth.shape
  x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
  x = x.view(1, 1, H, W).repeat(B, 1, 1, 1)
  y = y.view(1, 1, H, W).repeat(B, 1, 1, 1)

  depth = depth.squeeze(1)
  points_3d = torch.stack((x * depth * camera_matrix[..., 0, 0] + camera_matrix[..., 0, 2],
                         y * depth * camera_matrix[..., 1, 1] + camera_matrix[..., 1, 2],
                         depth * camera_matrix[..., 2, 2]), dim=1)
  
  if normalize:
    points_3d = points_3d / points_3d.norm(dim=1, keepdim=True)

  return points_3d

if __name__ == "__main__":
  B = 1
  H = 64
  W = 64
  depth = torch.randn(B, 1, H, W)
  camera_matrix = torch.randn(B, 3, 3)
  points_3d = depth_to_3d(depth, camera_matrix)
  print(f"Input depth shape: {depth.shape}")
  print(f"Camera matrix shape: {camera_matrix.shape}")
  print(f"Output points_3d shape: {points_3d.shape}")