import torch

def silu(input, inplace=False):
  if inplace:
    input.mul_(torch.sigmoid(input))
    return input
  else:
    return input * torch.sigmoid(input)

if __name__ == "__main__":
  x = torch.randn(3, 4)
  print(f"Input Tensor:\n{x}")
  y = silu(x)
  print(f"Output Tensor:\n{y}")