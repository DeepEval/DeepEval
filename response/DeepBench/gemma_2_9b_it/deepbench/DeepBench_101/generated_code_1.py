import torch

def su2_generators(k):
  j = k // 2
  x = torch.tensor([[0,  sqrt(j*(j+1)), 0],
                   [sqrt(j*(j+1)), 0, sqrt(j*(j+1))],
                   [0, sqrt(j*(j+1)), 0]], dtype=torch.float32)
  y = torch.tensor([[0, -1j*sqrt(j*(j+1)), 0],
                   [1j*sqrt(j*(j+1)), 0, -1j*sqrt(j*(j+1))],
                   [0, 1j*sqrt(j*(j+1)), 0]], dtype=torch.float32)
  z = torch.tensor([[j, 0, 0],
                   [0, 0, 0],
                   [0, 0, -j]], dtype=torch.float32)
  return torch.stack((x, z, y))

if __name__ == "__main__":
  k = 2
  generators = su2_generators(k)
  print(generators)