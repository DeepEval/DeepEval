import numpy as np
import opt_einsum as oe

def einsum(equation, *operands):
    if len(operands) >= 3:
        return oe.contract(equation, *operands, optimize='optimal')
    else:
        return np.einsum(equation, *operands)

if __name__ == "__main__":
    # Example usage
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[9, 10], [11, 12]])
    
    # Einstein summation convention: sum over all axes (i,j) of A*B
    result = einsum('ij,ij->', A, B)
    print("Result of A*B:", result)
    
    # Example with broadcasting
    result = einsum('ij,j->i', A, B)
    print("Result of A*B with broadcasting:", result)
    
    # Example with ellipsis for adding dimensions
    D = np.array([13, 14])
    result = einsum('...j,j->...', A, D)
    print("Result of A*D with ellipsis:", result)