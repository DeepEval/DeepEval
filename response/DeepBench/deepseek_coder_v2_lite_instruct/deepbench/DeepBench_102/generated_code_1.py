import torch

def wigner_D(k, alpha, beta, gamma):
    # Initialize the Wigner D matrix
    D = torch.zeros((len(alpha), 2*k+1, 2*k+1))
    
    # Iterate over each rotation angle
    for i, (a, b, g) in enumerate(zip(alpha, beta, gamma)):
        D[i] = wigner_D_element(k, a, b, g)
    
    return D

def wigner_D_element(k, alpha, beta, gamma):
    # Compute the Wigner D matrix element
    D = 0
    for m in range(-k, k+1):
        for n in range(-k, k+1):
            D += torch.exp(1j * (m * alpha + n * beta + m * gamma)) * \
                 wigner_d_matrix(k, m, n, beta)
    return D

def wigner_d_matrix(k, m, n, beta):
    # Compute the Wigner d matrix element
    if m == n:
        return (-1)**m * torch.cos(beta/2)**(2*m) * torch.sin(beta/2)**2
    else:
        return (-1)**m * torch.sin(beta/2) * (torch.cos(beta/2)**(2*m) * torch.cos(beta/2)**(2*n) - \
                                             torch.sin(beta/2)**(2*m) * torch.sin(beta/2)**(2*n))

if __name__ == "__main__":
    k = 2
    alpha = torch.tensor([0, 0, 0])
    beta = torch.tensor([0, 0, 0])
    gamma = torch.tensor([0, 0, 0])
    
    result = wigner_D(k, alpha, beta, gamma)
    print(result)