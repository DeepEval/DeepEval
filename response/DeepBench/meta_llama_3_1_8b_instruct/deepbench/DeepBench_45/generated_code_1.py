import torch
import numpy as np

def change_basis_real_to_complex(k, dtype=None, device=None):
    # Generate a 2D tensor to store the transformation matrix Q
    Q = torch.zeros((2 * k + 1) ** 2, (2 * k + 1) ** 2, dtype=dtype, device=device)
    
    # Populate the transformation matrix Q based on the relationship between real and complex spherical harmonics
    for l in range(k + 1):
        for m in range(-l, l + 1):
            for l_prime in range(k + 1):
                for m_prime in range(-l_prime, l_prime + 1):
                    if m_prime == m:
                        if m >= 0:
                            Q[l + k * (l + 1), l_prime + k * (l_prime + 1)] = 1 / np.sqrt(2)
                            Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = -1j / np.sqrt(2)
                        else:
                            Q[l + k * (l + 1), l_prime + k * (l_prime + 1)] = 1 / np.sqrt(2)
                            Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = 1j / np.sqrt(2)
                    elif m_prime == m + 1:
                        if m >= 0:
                            Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = 1j
                        else:
                            Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = -1j
                    elif m_prime == m - 1:
                        if m >= 0:
                            Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = 1j
                        else:
                            Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = -1j
                    else:
                        Q[l + k * (l + 1), l_prime + k * (l_prime + 1)] = 0
                        Q[l + k * (l + 1) + (l_prime + k * (l_prime + 1)) * (2 * k + 1), l_prime + k * (l_prime + 1)] = 0
    
    return Q

if __name__ == "__main__":
    k = 3
    Q = change_basis_real_to_complex(k)
    print(Q)