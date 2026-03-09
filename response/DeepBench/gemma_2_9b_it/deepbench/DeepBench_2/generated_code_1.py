import numpy as np

def apply_similarity_transformation(X, R, T, s):
  transformed_X = np.einsum('bij,ik->bik', R, X)
  transformed_X += T[:, None, :]
  transformed_X *= s[:, None, None]
  return transformed_X

if __name__ == "__main__":
  batch_size = 2
  num_points = 10
  d = 3
  
  X = np.random.randn(batch_size, num_points, d)
  R = np.random.randn(batch_size, d, d)
  R = R @ R.transpose(0,2,1)  # Ensure orthonormality
  T = np.random.randn(batch_size, d)
  s = np.random.randn(batch_size)

  transformed_X = apply_similarity_transformation(X, R, T, s)
  print("Original X shape:", X.shape)
  print("Transformed X shape:", transformed_X.shape)