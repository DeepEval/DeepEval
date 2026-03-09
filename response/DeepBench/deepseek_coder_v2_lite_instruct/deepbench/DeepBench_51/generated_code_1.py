import numpy as np
import scipy.linalg as la

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    L = q_mu.shape[1]
    M = q_mu.shape[0]

    if K is not None:
        if K_cholesky is not None:
            raise ValueError("Cannot specify both K and K_cholesky")
        K = K
    elif K_cholesky is not None:
        K = K_cholesky @ K_cholesky.T
    else:
        K = np.eye(M)

    if q_sqrt.ndim == 3:
        q_sqrt = q_sqrt.reshape(L, M, M)
        log_det_q_sqrt = np.sum(np.log(np.diagonal(q_sqrt, axis1=-2, axis2=-1)), axis=1)
        inv_q_sqrt = np.linalg.inv(q_sqrt)
    elif q_sqrt.ndim == 2:
        log_det_q_sqrt = 2 * np.sum(np.log(np.abs(np.diag(q_sqrt))), axis=1)
        inv_q_sqrt = np.diag(1.0 / np.diag(q_sqrt))
    else:
        raise ValueError("q_sqrt must be 2D or 3D")

    trace_term = np.sum(np.einsum('ml,nml->n', inv_q_sqrt, np.linalg.inv(K)), axis=0)
    mean_term = np.sum(np.einsum('nm,nm->n', q_mu, q_mu) - 2 * np.einsum('nm,nm->n', q_mu, np.linalg.solve(K, q_mu)), axis=0)

    kl = 0.5 * (trace_term + mean_term - M + log_det_q_sqrt - np.sum(np.log(np.diagonal(K)), axis=0))

    return np.sum(kl)

if __name__ == "__main__":
    q_mu = np.array([[1, 2], [3, 4]])
    q_sqrt_3d = np.array([[[2, 0], [0, 2]], [[3, 0], [0, 3]]])
    q_sqrt_2d = np.array([[2, 0], [0, 2]])
    K = np.array([[4, 1], [1, 4]])

    print("KL Divergence with 3D q_sqrt:", gauss_kl(q_mu, q_sqrt_3d, K))
    print("KL Divergence with 2D q_sqrt:", gauss_kl(q_mu, q_sqrt_2d, K))
    print("KL Divergence without K:", gauss_kl(q_mu, q_sqrt_2d))