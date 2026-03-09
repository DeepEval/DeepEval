import torch
import torch.nn as nn
import numpy as np

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    """
    Computes the KL divergence KL[q || p] between q(x) = N(q_mu, q_sqrt^2) and p(x) = N(0, K)
    """
    if K is None:
        K = np.eye(q_mu.shape[1])  # Use the identity matrix if no covariance K is given

    if q_sqrt.shape[1] == 1:
        q_sqrt = q_sqrt.squeeze(1)

    # Compute the trace of the product of q_sqrt and its transpose
    trace_product = torch.sum(torch.sum(q_sqrt, dim=1, keepdim=True)**2)

    # Compute the trace of the inverse of K
    trace_K_inv = torch.trace(1/K)

    # Compute the determinant of the covariance of q
    det_q_cov = torch.det(q_mu.matmul(q_sqrt).matmul(q_sqrt.t().matmul(q_mu.t())))

    # Compute the determinant of the covariance of p
    det_p_cov = torch.det(K)

    # Compute the KL divergence
    kl_div = trace_product + det_q_cov - det_p_cov + trace_K_inv
    return kl_div

if __name__ == "__main__":
    # Sample input values
    M = 2
    L = 3
    q_mu = torch.randn(M, L)
    q_sqrt = torch.randn(L, M, M)

    # K is the covariance of p
    K = torch.eye(M)

    # Compute KL divergence
    kl_div = gauss_kl(q_mu, q_sqrt, K)

    print("KL Divergence:", kl_div)