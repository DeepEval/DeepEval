import torch

def quaternion_to_axis_angle(quaternion):
  if isinstance(quaternion, torch.Tensor) and (quaternion.shape == torch.Size([None, 4]) or quaternion.shape == torch.Size([4])):
    w, x, y, z = quaternion.chunk(4)
    axis = torch.stack((x, y, z), dim=-1)
    angle = 2 * torch.acos(w)
    return torch.cat((axis, angle.unsqueeze(-1)), dim=-1)
  else:
    raise TypeError("Input must be a PyTorch tensor with shape Nx4 or 4")

if __name__ == "__main__":
  quaternion = torch.tensor([0.7071, 0.7071, 0, 0])
  axis_angle = quaternion_to_axis_angle(quaternion)
  print(axis_angle)