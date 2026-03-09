import numpy as np
from opt_einsum import contract

def einsum(equation, *operands):
    if len(operands) < 3:
        return np.sum(operands, axis=None)
    else:
        return contract(equation, *operands)

if __name__ == "__main__":
    # Example usage
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(4).reshape(1, 1, 4)
    c = np.arange(16).reshape(2, 2, 2, 2)

    # Sum the product of the elements of a and b along the last dimension
    result = einsum("ijk,k->ij", a, b)
    print(result)

    # Sum the product of the elements of a, b, and c along the last two dimensions
    result = einsum("ijk,k,lkl->ijl", a, b, c)
    print(result)

    # Sum the product of the elements of a and b along the last dimension, and the product of the elements of c along the last dimension
    result = einsum("ijk,k,l->ijl", a, b, c)
    print(result)