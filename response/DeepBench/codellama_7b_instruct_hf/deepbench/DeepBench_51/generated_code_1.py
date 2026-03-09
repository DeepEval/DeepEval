import numpy as np

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    """
    Computes the KL divergence KL[q || p] between q(x) = N(q_mu, q_sqrt^2) and p(x) = N(0, K) if K is not None, p(x) = N(0, I) if K is None.

    Args:
        q_mu (np.ndarray): mean of the Gaussian distribution q, of shape (M, L)
        q_sqrt (np.ndarray): lower triangular square-root matrix of the covariance of q, of shape (L, M, M) or (M, L)
        K (np.ndarray): covariance of the Gaussian distribution p, of shape (M, M) or (L, M, M)
        K_cholesky (np.ndarray): Cholesky factor of the covariance of p, of shape (M, M) or (L, M, M)

    Returns:
        output (np.ndarray): the sum of the KL divergences between q and p, of shape (L,)
    """

    # Check if K is passed as a Cholesky factor
    if K_cholesky is not None:
        K = np.matmul(K_cholesky, K_cholesky.transpose())

    # Compute the determinant of the covariance matrix
    det_q = np.linalg.det(q_sqrt)
    det_p = np.linalg.det(K)

    # Compute the logarithm of the determinant of the ratio of the covariance matrices
    log_det_ratio = np.log(det_q / det_p)

    # Compute the trace of the ratio of the covariance matrices
    trace_ratio = np.trace(q_sqrt @ np.linalg.inv(K))

    # Compute the KL divergence
    kl_div = 0.5 * (log_det_ratio + trace_ratio - M + np.trace(q_sqrt @ np.linalg.inv(K)))

    # Compute the sum of the KL divergences
    output = np.sum(kl_div, axis=0)

    return output

if __name__ == "__main__":
    # Example usage
    q_mu = np.random.randn(5, 2)
    q_sqrt = np.random.randn(2, 5, 5)
    K = np.random.randn(5, 5)
    K_cholesky = np.linalg.cholesky(K)
    output = gauss_kl(q_mu, q_sqrt, K, K_cholesky)
    print(output)