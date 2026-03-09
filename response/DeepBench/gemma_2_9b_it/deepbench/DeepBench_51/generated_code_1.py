import numpy as np

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
  M = q_mu.shape[0]
  L = q_mu.shape[1]
  if K is not None:
    if K.ndim == 2:
      K = np.expand_dims(K, axis=0).repeat(L, axis=0)
    
    K_inv = np.linalg.inv(K)
    K_sqrt_inv = np.linalg.inv(np.sqrt(K))
    
  elif K_cholesky is not None:
    if K_cholesky.ndim == 2:
      K_cholesky = np.expand_dims(K_cholesky, axis=0).repeat(L, axis=0)
    
    K_inv = np.linalg.inv(np.matmul(K_cholesky, K_cholesky.transpose(0,2,1)))
    K_sqrt_inv = np.linalg.inv(np.sqrt(np.matmul(K_cholesky, K_cholesky.transpose(0,2,1))))
  else:
    K_inv = np.eye(M)
    K_sqrt_inv = np.eye(M)
  
  if q_sqrt.ndim == 3:
    q_sqrt_inv = np.linalg.inv(q_sqrt)
  elif q_sqrt.ndim == 2:
    q_sqrt_inv = np.diag(1.0 / np.sqrt(q_sqrt))
  else:
    raise ValueError("q_sqrt must be a 2D or 3D tensor.")

  kl_div = 0.5 * np.sum(
      np.log(np.linalg.det(K_sqrt_inv)) - np.log(np.linalg.det(q_sqrt_inv))
      + np.trace(np.matmul(K_inv, q_sqrt))
      - M
  , axis=1)
  return np.sum(kl_div)

if __name__ == "__main__":
  q_mu = np.random.randn(3, 2)
  if q_sqrt.ndim == 3:
    q_sqrt = np.random.randn(2, 3, 3)
  elif q_sqrt.ndim == 2:
    q_sqrt = np.random.randn(3, 2)
  else:
    raise ValueError("q_sqrt must be a 2D or 3D tensor.")
  K = np.random.randn(3, 3)
  kl = gauss_kl(q_mu, q_sqrt, K=K)
  print(f'KL divergence: {kl}')