import torch

def quaternion_exp_to_log(quaternion, eps=1e-6):
  if isinstance(quaternion, torch.Tensor) and quaternion.shape == (*, 4):
    w, x, y, z = quaternion.unbind(dim=-1)
    norm = torch.norm(torch.stack((x, y, z), dim=-1), dim=-1)
    return torch.where(norm > eps, torch.stack((0.5 * torch.log(norm + torch.exp(norm)), x / norm, y / norm, z / norm), dim=-1), torch.stack((0, x, y, z), dim=-1))

if __name__ == "__main__":
  quaternion = torch.tensor([1.0, 0.2, 0.3, 0.4])
  result = quaternion_exp_to_log(quaternion)
  print(result)