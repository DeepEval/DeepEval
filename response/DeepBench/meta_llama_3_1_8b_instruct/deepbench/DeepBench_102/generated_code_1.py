import torch
import torch.fft
import torch.linalg

def winger_D(k, alpha, beta, gamma):
    # Calculate the matrix elements
    a = torch.exp(1j * gamma)
    b = torch.exp(1j * (alpha + beta))
    c = torch.exp(-1j * (alpha - beta))
    d = torch.exp(1j * gamma)

    # Initialize the Wigner D matrix
    D = torch.zeros((len(alpha), 2*k + 1, 2*k + 1), dtype=torch.complex128)

    # Populate the Wigner D matrix
    for m in range(-k, k + 1):
        for n in range(-k, k + 1):
            D[:, m + k, n + k] = (
                a * b * c * d
                * (
                    (-1)**(-m + n)
                    * torch.complex_(torch.math.factorial(k + n) / (torch.math.factorial(k + m) * torch.math.factorial(k - m)))
                    * torch.complex_(torch.math.factorial(k - n) / (torch.math.factorial(k - m) * torch.math.factorial(k + m)))
                    * torch.complex_(torch.math.factorial(k + n) / (torch.math.factorial(k - n) * torch.math.factorial(k + n)))
                )
            )

    return D

if __name__ == "__main__":
    k = 2
    alpha = torch.tensor([0.1, 0.2, 0.3])
    beta = torch.tensor([0.4, 0.5, 0.6])
    gamma = torch.tensor([0.7, 0.8, 0.9])

    D = winger_D(k, alpha, beta, gamma)
    print(D)