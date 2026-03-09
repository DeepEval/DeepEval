import numpy as np
from scipy.linalg import cholesky, solve_triangular

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    M, L = q_mu.shape
    
    if K is not None:
        if K_cholesky is None:
            K_chol = cholesky(K, lower=True)
        else:
            K_chol = K_cholesky
        K_inv = solve_triangular(K_chol, np.eye(M), lower=True)
    else:
        K_chol = np.eye(M)
        K_inv = np.eye(M)

    if q_sqrt.ndim == 3:
        # Lower triangular case
        q_sqrt_inv = np.linalg.inv(q_sqrt)
        KL = 0.5 * (np.sum(np.square(q_mu), axis=0) - np.sum(np.log(np.diagonal(q_sqrt, axis1=1, axis2=2)), axis=0)
                    + np.sum(np.square(q_sqrt_inv @ K_chol), axis=(1, 2)) + M)
    else:
        # Diagonal case
        q_sqrt_diag = np.diagonal(q_sqrt, axis1=0, axis2=1)
        KL = 0.5 * (np.sum(np.square(q_mu), axis=0) - np.sum(np.log(q_sqrt_diag), axis=0)
                    + np.sum(np.square(K_chol @ q_sqrt_diag), axis=0) + M)

    return np.sum(KL)

if __name__ == "__main__":
    M = 3  # Number of dimensions
    L = 2  # Number of independent distributions

    q_mu = np.random.rand(M, L)
    q_sqrt = np.random.rand(M, L)  # Diagonal case
    K = np.random.rand(M, M)
    K = K @ K.T  # Make K positive definite

    kl_divergence = gauss_kl(q_mu, q_sqrt, K=K)
    print("KL Divergence:", kl_divergence)