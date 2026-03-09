import torch

def gauss_kl(q_mu, q_sqrt, K=None, K_cholesky=None):
    """Computes the KL divergence KL[q || p] between two normal distributions.

    q(x) = N(q_mu, q_sqrt^2)
    p(x) = N(0, K) if K is not None
    p(x) = N(0, I) if K is None

    We assume L multiple independent distributions, given by the columns of q_mu and the first or last dimension of q_sqrt.

    Returns the sum of the divergences.

    q_mu: Matrix ([M, L]), each column contains a mean.
    q_sqrt: 3D tensor ([L, M, M]) or matrix ([M, L]), each column represents the diagonal of a square-root matrix of the covariance of q.
    K: Covariance of p (positive-definite matrix). Can be passed either directly as K, or as its Cholesky factor, K_cholesky.
    K_cholesky: Cholesky factor of the covariance of p. Can be passed either directly as K_cholesky, or as K.
    """

    # Set default values for K and K_cholesky if not provided
    if K is None and K_cholesky is None:
        K = torch.eye(q_mu.shape[0])

    # Calculate the KL divergence for each distribution
    kl_div = 0.5 * (
        (q_mu ** 2).sum(dim=1)
        - (K ** 2).sum(dim=(0, 1))
        - torch.log(
            torch.tensor(2 * torch.pi) ** q_mu.shape[0] * torch.det(K) / torch.det(q_sqrt)
        )
    )

    # Sum the divergences and return the result
    return kl_div.sum()


if __name__ == "__main__":
    # Create sample input values
    q_mu = torch.randn(5, 2)
    q_sqrt = torch.randn(2, 5, 5)
    K = torch.eye(5)

    # Call the function and print the results
    kl_div = gauss_kl(q_mu, q_sqrt, K)
    print(kl_div)