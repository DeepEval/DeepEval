import numpy as np

def find_homography_lines_dlt(ls1, ls2, weights=None):
  B, N, _, _ = ls1.shape
  A = np.zeros((N * 4, 9))
  b = np.zeros((N * 4, 1))
  for i in range(N):
    l1 = ls1[0, i]
    l2 = ls2[0, i]
    A[i * 4:i * 4 + 4, :] = np.array([
      [l1[0], l1[1], 1, 0, 0, 0, -l2[0] * l1[0], -l2[0] * l1[1], -l2[0]],
      [0, 0, 0, l1[0], l1[1], 1, -l2[1] * l1[0], -l2[1] * l1[1], -l2[1]]
    ])
    if weights is not None:
      w = weights[0, i]
      b[i * 4:i * 4 + 4] = w * np.array([l2[0], l2[1]])
    else:
      b[i * 4:i * 4 + 4] = np.array([l2[0], l2[1]])
  H = np.linalg.lstsq(A, b, rcond=None)[0]
  return H.reshape(B, 3, 3)

if __name__ == "__main__":
  B = 1
  N = 2
  ls1 = np.random.rand(B, N, 2, 2)
  ls2 = np.random.rand(B, N, 2, 2)
  weights = np.random.rand(B, N)
  H = find_homography_lines_dlt(ls1, ls2, weights)
  print(H)