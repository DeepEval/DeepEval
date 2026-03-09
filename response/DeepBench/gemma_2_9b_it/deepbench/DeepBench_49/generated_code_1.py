import torch
from torch import matrix_power

def matrix_cofactor_tensor(matrix):
  determinants = torch.det(matrix)
  if torch.all(determinants == 0):
    raise Exception("All input matrices are singular.")
  
  adj_matrix = matrix_power(matrix.transpose(-1, -2), -1)
  return adj_matrix * determinants[..., None, None]

if __name__ == "__main__":
  matrix = torch.tensor([
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],

    [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]]
  ])
  cofactor_matrix = matrix_cofactor_tensor(matrix)
  print(cofactor_matrix)