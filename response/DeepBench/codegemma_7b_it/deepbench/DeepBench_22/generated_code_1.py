import opt_einsum
import numpy as np

def einsum(equation, *operands):
    return opt_einsum.einsum(equation, *operands)

if __name__ == "__main__":
    # Example 1:
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(12).reshape(3, 4)
    c = einsum("ijk,kl->ij", a, b)
    print(c)

    # Example 2:
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(12).reshape(4, 3)
    c = einsum("ijk,kl->ij", a, b)
    print(c)

    # Example 3:
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(24).reshape(4, 3, 2)
    c = einsum("ijk,klm->ijm", a, b)
    print(c)

    # Example 4:
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(24).reshape(2, 4, 3)
    c = einsum("ijk,klm->ijm", a, b)
    print(c)