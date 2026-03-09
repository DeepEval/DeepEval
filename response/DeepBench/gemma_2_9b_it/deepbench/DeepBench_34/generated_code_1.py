import numpy as np

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
  if E_mat.shape != (3, 3):
    raise ValueError("Essential matrix must have shape (3, 3)")
  if K1.shape != (3, 3) or K2.shape != (3, 3):
    raise ValueError("Camera matrices must have shape (3, 3)")
  if x1.shape != (..., 2) or x2.shape != (..., 2):
    raise ValueError("Point sets must have shape (..., 2)")
  
  U, S, Vt = np.linalg.svd(E_mat)
  R1 = Vt.T @ U.T
  R2 = -Vt.T @ U.T

  t1 = Vt[:, 2]
  t2 = -R1 @ t1

  points_3d = []
  for i in range(x1.shape[0]):
    if mask is not None and mask[i] == False:
      continue
    x1_ = K1 @ x1[i]
    x2_ = K2 @ x2[i]
    A = np.array([
      [x1_[0], x1_[1], 1],
      [x2_[0], x2_[1], 1]
    ])
    
    b = np.array([
      [0, 0, 0],
      [0, 0, 0]
    ])
    
    
    
    
  return R1, t1, points_3d

if __name__ == "__main__":
  E_mat = np.random.rand(3, 3)
  K1 = np.eye(3)
  K2 = np.eye(3)
  x1 = np.random.rand(10, 2)
  x2 = np.random.rand(10, 2)
  R, t, points_3d = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
  print("Rotation Matrix:\n", R)
  print("Translation Vector:\n", t)
  print("3D Points:\n", points_3d)