import torch
import numpy as np

def winger_D(k, alpha, beta, gamma):
    # Number of output angles
    num_angles = alpha.shape[0]
    
    # Initialize the Wigner D matrix
    D = torch.zeros((num_angles, 2*k + 1, 2*k + 1), dtype=torch.complex128)
    
    # Create the rotation matrices
    for i in range(num_angles):
        c_alpha = torch.cos(alpha[i])
        s_alpha = torch.sin(alpha[i])
        c_beta = torch.cos(beta[i])
        s_beta = torch.sin(beta[i])
        c_gamma = torch.cos(gamma[i])
        s_gamma = torch.sin(gamma[i])
        
        R = torch.tensor([[c_gamma * c_alpha - s_gamma * c_beta * s_alpha, -s_gamma * c_alpha - c_gamma * s_beta * s_alpha, s_beta * s_alpha],
                          [s_gamma * c_alpha + c_gamma * s_beta * s_alpha, c_gamma * c_alpha * s_beta - s_gamma * s_alpha, -c_beta * s_alpha],
                          [s_beta * s_alpha, c_beta * s_alpha, c_alpha]])
        
        # Compute the Wigner D matrix for the current angle
        for m in range(-k, k + 1):
            for n in range(-k, k + 1):
                D[i, m + k, n + k] = (1j)**(m - n) * torch.sqrt((2 * k + 1) * torch.factorial(k + m) * torch.factorial(k - m) * torch.factorial(k + n) * torch.factorial(k - n)) \
                                      * (1 / (2 * np.pi)) * (torch.tensor(np.exp(1j * (m * alpha[i] + n * gamma[i])))) * R[m + k, n + k])
    
    return D

if __name__ == "__main__":
    k = 1
    alpha = torch.tensor([0.1, 0.2])
    beta = torch.tensor([0.1, 0.2])
    gamma = torch.tensor([0.1, 0.2])
    
    D_matrix = winger_D(k, alpha, beta, gamma)
    print(D_matrix)