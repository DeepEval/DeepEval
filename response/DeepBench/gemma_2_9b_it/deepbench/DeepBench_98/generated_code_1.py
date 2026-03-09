import torch
import numpy as np

def run_8point(points1, points2, weights):
  B, N, _ = points1.shape
  A = torch.zeros((B, N, 9))
  for i in range(B):
    for j in range(N):
      x1, y1 = points1[i, j]
      x2, y2 = points2[i, j]
      A[i, j * 3:j * 3 + 3] = torch.tensor([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
  
  A = A.reshape(B * N, 9)
  weights = weights.reshape(B * N)
  
  X = torch.linalg.solve(A.t() @ torch.diag(weights) @ A, A.t() @ torch.diag(weights) @ torch.cat((points1.reshape(B * N, 2), points2.reshape(B * N, 2)), dim=1).reshape(B * N, 4))
  
  F = X.reshape(B, 3, 3)
  return F

if __name__ == "__main__":
  points1 = torch.randn(2, 8, 2)
  points2 = torch.randn(2, 8, 2)
  weights = torch.ones(2, 8)
  F = run_8point(points1, points2, weights)
  print(F)