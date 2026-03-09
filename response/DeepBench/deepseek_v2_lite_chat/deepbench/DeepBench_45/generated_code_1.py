import torch
import numpy as np

def change_basis_real_to_complex(k, dtype=None, device=None):
    # Compute the normalization constant for the real spherical harmonics
    normalization_constant = 2 * np.pi * np.random.random()
    
    # Define the real spherical harmonics basis functions
    real_spherical_harmonics = np.random.random((k, k))
    
    # Calculate the spherical harmonics matrix
    Y_matrix = np.zeros((2 * k + 1, 2 * k + 1))
    Y_matrix[0, 0] = 1
    for l in range(1, k + 1):
        for m in (-l, l):
            Y_matrix[l + 1, l - abs(m)] = (
                (-1) ** m * np.sqrt((2 * l + 1) / (4 * np.pi) *
                                    (l - abs(m)) * (l + abs(m) + 1)) *
                real_spherical_harmonics[l - 1, m])
    
    # Compute the complex spherical harmonics matrix
    complex_spherical_harmonics = np.zeros((2 * k + 1, 2 * k + 1))
    for l in range(k + 1):
        for m in (-l, l):
            complex_spherical_harmonics[l + 1, l - abs(m)] = (
                Y_matrix[l, l - abs(m)] +
                1j * Y_matrix[l, l - abs(m) + 1])
    
    # Normalize the complex spherical harmonics matrix
    for l in range(k + 1):
        for m in (-l, l):
            complex_spherical_harmonics[l, l - abs(m)] /= normalization_constant
            complex_spherical_harmonics[l, l - abs(m) + 1] /= normalization_constant
    
    # Construct the transformation matrix Q
    Q = torch.zeros((2 * k + 1, 2 * k + 1), dtype=dtype, device=device)
    for l in range(k + 1):
        for m in (-l, l):
            Q[l + 1, l - abs(m)] = complex_spherical_harmonics[l, l - abs(m)]
            Q[l + 1, l - abs(m) + 1] = complex_spherical_harmonics[l, l - abs(m) + 1]
    
    return Q

if __name__ == "__main__":
    # Example usage
    k = 3
    Q = change_basis_real_to_complex(k)
    print("Transformation matrix Q:\n", Q)