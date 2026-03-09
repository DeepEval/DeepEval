import numpy as np
from numpy.linalg import svd

def run_7point(points1, points2):
  assert points1.shape == (None, 7, 2) and points2.shape == (None, 7, 2)
  B, N, _ = points1.shape
  points1 = (points1 - points1.mean(axis=1, keepdims=True)) / points1.std(axis=1, keepdims=True)
  points2 = (points2 - points2.mean(axis=1, keepdims=True)) / points2.std(axis=1, keepdims=True)

  A = np.zeros((2*N, 9))
  for i in range(N):
    x1, y1 = points1[:, i]
    x2, y2 = points2[:, i]
    A[2*i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    A[2*i+1] = [-y1*x2, -y1*y2, -y1, x1*x2, x1*y2, x1, -x2, -y2, -1]

  U, S, Vt = svd(A)
  F = Vt[-1].reshape(3, 3)
  F = F / F[2, 2]
  
  U, S, Vt = svd(F)
  if np.linalg.det(U) < 0:
    Vt[-1] *= -1
    F = Vt.T @ U.T
  
  F = F / F[2, 2]

  # Generate potential matrices
  F_candidates = []
  for i in range(3):
    if i == 0:
      F_candidate = F
    else:
      F_candidate = np.array([
        [F[0, 0], F[0, 1], F[0, 2]],
        [F[1, 0], F[1, 1], F[1, 2]],
        [F[2, 0], F[2, 1], -F[2, 2]]
      ])
    F_candidates.append(F_candidate)
  return np.array(F_candidates)



if __name__ == "__main__":
  points1 = np.random.rand(1, 7, 2)
  points2 = np.random.rand(1, 7, 2)
  fundamental_matrices = run_7point(points1, points2)
  print(fundamental_matrices)