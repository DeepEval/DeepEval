import numpy as np
from opt_einsum import contract

def einsum(equation, *operands):
    return contract(equation, *operands)

if __name__ == "__main__":
    A = np.random.rand(2, 3)  # A 2x3 matrix
    B = np.random.rand(3, 4)  # A 3x4 matrix
    C = np.random.rand(4, 2)  # A 4x2 matrix

    result = einsum('ij,jk,kl->il', A, B, C)
    print(result)