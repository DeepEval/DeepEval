import torch

def pyramid_combination(values, weight_floor, weight_ceil):
    if len(values) != 2**len(weight_floor):
      raise ValueError("The length of 'values' must be 2^n, where n is the length of 'weight_floor' and 'weight_ceil'.")
    if len(weight_floor) != len(weight_ceil):
      raise ValueError("The lengths of 'weight_floor' and 'weight_ceil' must be equal.")
    for i in range(len(values)):
      if values[i].shape != weight_floor[0].shape or values[i].shape != weight_ceil[0].shape:
        raise ValueError("The shapes of 'values', 'weight_floor', and 'weight_ceil' must be compatible.")
    
    n = len(weight_floor)
    combined = torch.zeros_like(values[0])
    for i in range(2**n):
      weight = torch.prod(torch.stack([weight_floor[j][i//2**(n-j-1)] for j in range(n)]), dim=0) * torch.prod(torch.stack([weight_ceil[j][i//2**(n-j-1)] for j in range(n)]), dim=0)
      combined += weight * values[i]
    return combined

if __name__ == "__main__":
    values = [torch.randn(2) for _ in range(8)]
    weight_floor = [torch.randn(2) for _ in range(3)]
    weight_ceil = [torch.randn(2) for _ in range(3)]
    result = pyramid_combination(values, weight_floor, weight_ceil)
    print(result)