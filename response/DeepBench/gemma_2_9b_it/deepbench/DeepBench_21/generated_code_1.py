import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
  if sigma <= 0.0:
    raise ValueError
  normalization_constant = torch.sum(torch.exp(-0.5 * (offsets ** 2) / sigma ** 2)).item()
  return (1 + 0.05) / normalization_constant

if __name__ == "__main__":
  offsets = torch.tensor([[1, 2, 3], [4, 5, 6]])
  sigma = 0.8
  result = _get_splat_kernel_normalization(offsets, sigma)
  print(f"Normalization constant: {result}")