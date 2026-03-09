import torch

def SO3_generators(k):
    """
    Generates the generators of the special orthogonal group SO(3).

    Args:
        k: int, the representation index.

    Returns:
        torch.Tensor, a stack of three SO(3) generators, corresponding to J_x, J_z, and J_y.
    """

    # Calculate the size of the representation
    dim = 2 * k

    # Create the generator matrices
    J_x = torch.zeros(dim, dim)
    J_z = torch.zeros(dim, dim)
    J_y = torch.zeros(dim, dim)

    # Fill in the matrices based on the representation index
    for i in range(k):
        J_x[i * 2, i * 2 + 1] = -1
        J_x[i * 2 + 1, i * 2] = 1

        J_z[i * 2, i * 2] = 1
        J_z[i * 2 + 1, i * 2 + 1] = -1

        J_y[i * 2, i * 2 + 1] = -1
        J_y[i * 2 + 1, i * 2] = 1

    # Create a stack of the generators
    generators = torch.stack([J_x, J_z, J_y])

    return generators

if __name__ == "__main__":
    # Create sample input values
    k = 2

    # Call the function and print the results
    generators = SO3_generators(k)
    print(generators)