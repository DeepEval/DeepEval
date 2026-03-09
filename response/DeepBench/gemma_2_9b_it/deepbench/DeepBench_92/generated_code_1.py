import torch

def KORNIA_CHECK_LAF(laf, raises=True):
  if laf.shape != torch.Size((laf.shape[0], laf.shape[1], 2, 3)):
    if raises:
      raise ValueError(f"Invalid LAF shape: {laf.shape}. Expected (B,N,2,3)")
    else:
      return False
  return True

if __name__ == "__main__":
  # Example usage
  laf_valid = torch.randn(2, 5, 2, 3)
  laf_invalid = torch.randn(2, 5, 3, 3)

  print(f"Valid LAF shape: {laf_valid.shape}, Check result: {KORNIA_CHECK_LAF(laf_valid)}")
  print(f"Invalid LAF shape: {laf_invalid.shape}, Check result: {KORNIA_CHECK_LAF(laf_invalid)}")