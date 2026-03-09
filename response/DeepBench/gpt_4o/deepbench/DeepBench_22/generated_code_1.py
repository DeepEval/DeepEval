import numpy as np
from opt_einsum import contract

def einsum(equation, *operands):
    num_inputs = len(operands)
    if num_inputs < 3:
        # Use numpy's einsum for less than three operands
        return np.einsum(equation, *operands)
    else:
        # Use opt_einsum for three or more operands
        return contract(equation, *operands)

if __name__ == "__main__":
    # Example usage
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[9, 10], [11, 12]])

    # Example for two operands
    result1 = einsum('ij,jk->ik', A, B)
    print("Result of A*B with equation 'ij,jk->ik':\n", result1)

    # Example for three operands
    result2 = einsum('ij,jk,kl->il', A, B, C)
    print("Result of A*B*C with equation 'ij,jk,kl->il':\n", result2)

    # Example with ellipsis
    D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    E = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    
    result3 = einsum('...ij,...jk->...ik', D, E)
    print("Result with ellipsis '...ij,...jk->...ik':\n", result3)