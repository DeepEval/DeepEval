import torch
from scipy.special import factorial
import numpy as np

def winger_D(k, alpha, beta, gamma):
    def little_d(k, m, n, beta):
        # Compute the little d matrix, which is a real-valued matrix
        # This uses the Wigner small d-matrix formula
        d = 0.0
        for s in range(max(0, m-n), min(k+m, k-n)+1):
            num = ((-1.0)**s) * factorial(k+m) * factorial(k-m) * factorial(k+n) * factorial(k-n)
            denom = factorial(k+m-s) * factorial(k-n-s) * factorial(s) * factorial(s+n-m)
            d += (num / denom) * (np.cos(beta/2)**(2*k+n-m-2*s)) * (np.sin(beta/2)**(m-n+2*s))
        return d

    alpha, beta, gamma = alpha.numpy(), beta.numpy(), gamma.numpy()
    
    # Precompute exponentials for alpha and gamma
    exp_alpha = np.exp(-1j * alpha)
    exp_gamma = np.exp(-1j * gamma)
    
    D_matrices = []
    for a, b, g in zip(exp_alpha, beta, exp_gamma):
        D = torch.zeros((2*k+1, 2*k+1), dtype=torch.complex128)
        for m in range(-k, k+1):
            for n in range(-k, k+1):
                d = little_d(k, m, n, b)
                D[m+k, n+k] = d * np.exp(-1j * m * a) * np.exp(-1j * n * g)
        D_matrices.append(D)

    return torch.stack(D_matrices)

if __name__ == "__main__":
    # Example usage of the winger_D function
    k = 1  # representation index
    alpha = torch.tensor([0.0, np.pi/2], dtype=torch.float64)  # rotation angles around Y axis, applied third
    beta = torch.tensor([np.pi/4, np.pi/3], dtype=torch.float64)  # rotation angles around X axis, applied second
    gamma = torch.tensor([np.pi/6, np.pi/4], dtype=torch.float64)  # rotation angles around Y axis, applied first
    
    D_matrix = winger_D(k, alpha, beta, gamma)
    print(D_matrix)