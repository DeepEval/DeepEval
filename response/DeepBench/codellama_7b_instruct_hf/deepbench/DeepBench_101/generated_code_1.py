import torch

def su2_generators(k):
    """
    Compute the generators of the SU(2) group for a specific representation
    determined by the value of 'k'.

    Parameters:
    k (int): The representation index, which determines the order of the representation.

    Returns:
    torch.Tensor: A stack of three SU(2) generators, corresponding to J_x, J_z, and J_y.
    """
    # Define the generators for the SU(2) group
    J_x = torch.tensor([[0, 0], [0, 1]])
    J_y = torch.tensor([[0, -1j], [1j, 0]])
    J_z = torch.tensor([[1, 0], [0, -1]])

    # Stack the generators in the correct order
    generators = torch.stack([J_x, J_y, J_z], dim=0)

    # Return the generators for the specified representation
    if k == 1:
        return generators
    else:
        return None

# Example usage
if __name__ == "__main__":
    # Generate the generators for the SU(2) group with k = 1
    generators = su2_generators(1)

    # Print the generators
    print(generators)