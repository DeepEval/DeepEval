import torch
import torch.nn as nn
import math

def su2_generators(k):
    # Define the dimensions of the matrix
    size = 2*k + 1

    # Initialize the matrices for J_x, J_y, and J_z
    J_x = torch.zeros((3, size, size))
    J_y = torch.zeros((3, size, size))
    J_z = torch.zeros((3, size, size))

    # Calculate the matrices for J_x, J_y, and J_z
    J_x[0, 1:-1, 1:-1] = torch.stack([torch.arange(-k, k+1)[:-1] for _ in range(size-1)])
    J_y[0, 1:-1, 1:-1] = torch.stack([torch.arange(-k, k+1)[1:] for _ in range(size-1)])
    J_z[0, 1:-1, 1:-1] = torch.eye(size-2)

    # Calculate the normalization factor
    norm = torch.sqrt(torch.sum(J_x[0, 1:-1, 1:-1]**2) + torch.sum(J_y[0, 1:-1, 1:-1]**2) + torch.sum(J_z[0, 1:-1, 1:-1]**2))

    # Normalize the matrices
    J_x[0, 1:-1, 1:-1] /= norm
    J_y[0, 1:-1, 1:-1] /= norm
    J_z[0, 1:-1, 1:-1] /= norm

    # Calculate the ladder operators J_plus and J_minus
    J_plus = (J_x + 1j*J_y)
    J_minus = (J_x - 1j*J_y)

    # Stack the matrices for J_x, J_y, J_z, J_plus, and J_minus
    generators = torch.stack([J_x, J_z, J_y, J_plus, J_minus])

    return generators

if __name__ == "__main__":
    k = 2
    generators = su2_generators(k)
    print(generators)