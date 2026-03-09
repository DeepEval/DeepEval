import torch

def so3_generators(k):
  from torch import tensor
  su2_generators = tensor([
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]
  ])
  so3_generators = (1/k) * (su2_generators + su2_generators.transpose(0, 1))
  return so3_generators

if __name__ == "__main__":
  k = 2
  generators = so3_generators(k)
  print(generators)