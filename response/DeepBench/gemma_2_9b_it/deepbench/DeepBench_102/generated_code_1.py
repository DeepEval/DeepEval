import torch
import numpy as np

def winger_D(k, alpha, beta, gamma):
    n = 2*k + 1
    a = torch.cos(alpha/2)
    b = torch.sin(alpha/2)
    c = torch.cos(beta/2)
    d = torch.sin(beta/2)
    e = torch.cos(gamma/2)
    f = torch.sin(gamma/2)

    matrix = torch.zeros(alpha.shape[0], n, n)
    for i in range(n):
      for j in range(n):
        matrix[:, i, j] =  (a**(abs(i-j))) * (b**(abs(i+j))) * (c**(i-j)) * (d**(i+j)) * (e**(i+j)) * (f**(i-j))

    return matrix

if __name__ == "__main__":
    k = 2
    alpha = torch.tensor([0, np.pi/4, np.pi/2])
    beta = torch.tensor([0, np.pi/3, np.pi/2])
    gamma = torch.tensor([0, np.pi/6, np.pi/4])
    
    D_matrix = winger_D(k, alpha, beta, gamma)
    print(D_matrix)