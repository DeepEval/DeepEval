import torch
import numpy as np

def change_basis_real_to_complex(k, dtype=None, device=None):
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cpu')
        
    l_max = k  # Maximum degree of spherical harmonics
    size = (l_max + 1) ** 2  # Total number of spherical harmonics
    Q = torch.zeros((size, size), dtype=dtype, device=device)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            index_real = l * (l + 1) + m  # Index for real harmonics
            if m >= 0:
                index_complex = l * (l + 1) + m  # Index for complex harmonics
                Q[index_real, index_complex] = np.sqrt(0.5)
                if m > 0:
                    Q[index_real, l * (l + 1) - m] = -np.sqrt(0.5) * 1j
            else:
                index_complex = l * (l + 1) - m  # Index for complex harmonics
                Q[index_real, index_complex] = np.sqrt(0.5)
                Q[index_real, l * (l + 1) + m] = np.sqrt(0.5) * 1j

    return Q

if __name__ == "__main__":
    k = 2
    transformation_matrix = change_basis_real_to_complex(k)
    print("Transformation Matrix Q from real to complex spherical harmonics:")
    print(transformation_matrix)