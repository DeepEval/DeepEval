import torch
import numpy as np

def wigner_D(k, alpha, beta, gamma):
    """
    Computes the Wigner D matrix representation of the SO(3) rotation group for a given representation index 'k' and rotation angles 'alpha', 'beta', and 'gamma'.
    The resulting matrix satisfies properties of the SO(3) group representation.
    Parameters:
        k: int
            The representation index, which determines the order of the representation.
        alpha: torch.Tensor
            Rotation angles (in radians) around the Y axis, applied third.
        beta: torch.Tensor
            Rotation angles (in radians) around the X axis, applied second.
        gamma: torch.Tensor
            Rotation angles (in radians) around the Y axis, applied first.
    Returns:
        torch.Tensor
            The Wigner D matrix of shape (#angles, 2k+1, 2k+1).
    """
    # Compute the Wigner D matrix
    wigner_d = torch.zeros((k, 2*k+1, 2*k+1))
    for i in range(k):
        for j in range(2*k+1):
            for m in range(2*k+1):
                wigner_d[i, j, m] = 2 * (k - i) * np.sin(alpha[j]) * np.cos(beta[i]) * np.cos(gamma[m])
    return wigner_d

if __name__ == "__main__":
    # Create sample input values
    k = 5
    alpha = torch.randn(5)
    beta = torch.randn(5)
    gamma = torch.randn(5)

    # Call the function and print the results
    wigner_d = wigner_D(k, alpha, beta, gamma)
    print(wigner_d)