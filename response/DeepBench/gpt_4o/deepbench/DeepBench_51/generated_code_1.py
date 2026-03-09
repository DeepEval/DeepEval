import numpy as np

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    M, L = q_mu.shape
    
    if q_sqrt.ndim == 3:
        q_cov = np.array([np.dot(q_sqrt[l], q_sqrt[l].T) for l in range(L)])
    else:
        q_cov = np.array([np.diag(q_sqrt[:, l] ** 2) for l in range(L)])
    
    if K is None and K_cholesky is None:
        K_cov = np.eye(M)
    elif K is not None:
        if K.ndim == 2:
            K_cov = np.broadcast_to(K, (L, M, M))
        else:
            K_cov = K
    else:
        if K_cholesky.ndim == 2:
            K_cov = np.array([np.dot(K_cholesky, K_cholesky.T)] * L)
        else:
            K_cov = np.array([np.dot(K_cholesky[l], K_cholesky[l].T) for l in range(L)])
    
    kl_div = 0
    for l in range(L):
        mu_term = np.dot(q_mu[:, l].T, np.dot(np.linalg.inv(K_cov[l]), q_mu[:, l]))
        trace_term = np.trace(np.dot(np.linalg.inv(K_cov[l]), q_cov[l]))
        log_det_q_cov = np.linalg.slogdet(q_cov[l])[1]
        log_det_k_cov = np.linalg.slogdet(K_cov[l])[1]
        kl_div += 0.5 * (trace_term + mu_term - M + log_det_k_cov - log_det_q_cov)
    
    return kl_div

if __name__ == "__main__":
    # Example inputs
    M, L = 3, 2
    q_mu = np.random.randn(M, L)
    q_sqrt = np.random.randn(L, M, M)  # assuming full covariance
    q_sqrt = np.array([np.tril(q_sqrt[l]) for l in range(L)])  # ensure lower triangular
    K = np.eye(M)

    # Calculate KL divergence
    kl_divergence = gauss_kl(q_mu, q_sqrt, K=K)
    print(f"KL divergence: {kl_divergence}")