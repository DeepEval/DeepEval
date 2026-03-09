import torch
import numpy as np

def so3_generators(k):
    """
    Generates the generators of the special orthogonal group SO(3)
    """
    # Generate the generators of the SU(2) group
    su2_generators = np.array([[0, 1, 0],
                               [0, 0, 1],
                               [1, 0, 0]])

    # Convert the SU(2) generators to the SO(3) basis
    so3_generators = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

    # Compute the SO(3) generators
    for i in range(3):
        for j in range(3):
            so3_generators[i, j] = su2_generators[i, j] * np.sqrt(k)

    return so3_generators

if __name__ == "__main__":
    # Example usage
    k = 2
    generators = so3_generators(k)
    print(generators)