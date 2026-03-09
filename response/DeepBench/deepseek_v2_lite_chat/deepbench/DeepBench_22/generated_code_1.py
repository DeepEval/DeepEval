import numpy as np
from numba import njit

@njit
def einsum(equation, *arrays):
    result = arrays[0]
    for i in range(1, len(arrays)):
        result = np.einsum(equation, *arrays[1:])
    return result

if __name__ == "__main__":
    # Sample input values
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = np.array([[9, 10], [11, 12]])
    d = np.array([[13, 14], [15, 16]])

    # Call the function and print the results
    result = einsum('ij,ik->ijk', a, b, c, d)
    print(result)