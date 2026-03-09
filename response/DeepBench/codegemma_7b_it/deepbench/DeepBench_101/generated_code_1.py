import torch
import math

def su2_generators(k):
    """Computes the generators of the SU(2) group for a specific representation.

    Args:
        k: int The representation index.

    Returns:
        torch.Tensor A stack of three SU(2) generators, corresponding to J_x, J_z, and J_y.
    """
    j = k / 2
    if k % 2 == 0:
        j = j - 0.5

    dim = 2 * int(j) + 1

    J_x = torch.zeros((dim, dim))
    J_z = torch.zeros((dim, dim))
    J_y = torch.zeros((dim, dim))

    for m in range(-int(j), int(j) + 1):
        if m == -int(j):
            J_x[m + int(j), m + int(j) + 1] = math.sqrt((j + m + 1) * (2 * j + 1))
            J_z[m + int(j), m + int(j)] = j + m
        elif m == int(j):
            J_x[m + int(j) - 1, m + int(j)] = math.sqrt((j + m + 1) * (2 * j + 1))
            J_z[m + int(j), m + int(j)] = j + m
        else:
            J_x[m + int(j), m + int(j) + 1] = math.sqrt((j + m + 1) * (j - m))
            J_x[m + int(j) - 1, m + int(j)] = math.sqrt((j + m + 1) * (j - m))
            J_z[m + int(j), m + int(j)] = j + m

    return torch.stack((J_x, J_z, J_y), dim=0)

if __name__ == "__main__":
    # Test the function with different representation indices
    k = 1
    generators = su2_generators(k)
    print(generators)

    k = 3
    generators = su2_generators(k)
    print(generators)