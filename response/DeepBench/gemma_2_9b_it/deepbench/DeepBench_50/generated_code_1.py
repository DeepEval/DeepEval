import torch

def aepe(input, target, reduction='mean'):
  diff = input - target
  err = torch.sqrt(torch.sum(diff**2, dim=-1))
  if reduction == 'mean':
    return torch.mean(err)
  elif reduction == 'sum':
    return torch.sum(err)
  else:
    return err

if __name__ == "__main__":
  input_flow = torch.randn(2, 10, 10, 2)
  target_flow = torch.randn(2, 10, 10, 2)
  aepe_value = aepe(input_flow, target_flow)
  print(f"AEPE: {aepe_value}")