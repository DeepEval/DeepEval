import numpy as np

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    M, L = q_mu.shape
    if q_sqrt.ndim == 3:
        q_sqrt = np.linalg.tril(q_sqrt, -1)
    elif q_sqrt.ndim == 2:
        q_sqrt = np.diag(q_sqrt)
    else:
        raise ValueError("q_sqrt must be a 2D or 3D tensor")
    
    if K is None and K_cholesky is None:
        # KL divergence from p(x) = N(0, I)
        KL = 0.5 * (M * np.log(2 * np.pi) + np.sum(np.diag(q_sqrt)**2))
    elif K is None:
        K = np.linalg.cholesky(K_cholesky)
    elif K_cholesky is None:
        K = K_cholesky
    
    if K.ndim == 2:
        # L different covariances
        KL = np.zeros(L)
        for i in range(L):
            inv_K = np.linalg.inv(K[:, :, i])
            diff = q_mu[:, i] - np.zeros(M)
            KL[i] = 0.5 * (np.dot(diff.T, inv_K @ diff) + np.sum(np.diag(q_sqrt[:, i])**2) + 
                           np.log(np.linalg.det(K[:, :, i])) - M)
    else:
        # single covariance matrix
        inv_K = np.linalg.inv(K)
        diff = q_mu - np.zeros(M)
        KL = 0.5 * (np.dot(diff.T, inv_K @ diff) + np.sum(np.diag(q_sqrt)**2) + 
                    np.log(np.linalg.det(K)) - M)
    
    return np.sum(KL)

if __name__ == "__main__":
    q_mu = np.array([[1, 2], [3, 4]])
    q_sqrt = np.array([[1, 0], [0, 2]])
    K = np.array([[1, 0.5], [0.5, 1]])
    K_cholesky = np.linalg.cholesky(K)
    print(gauss_kl(q_mu, q_sqrt, K=K))
    print(gauss_kl(q_mu, q_sqrt, K_cholesky=K_cholesky))
    print(gauss_kl(q_mu, q_sqrt))