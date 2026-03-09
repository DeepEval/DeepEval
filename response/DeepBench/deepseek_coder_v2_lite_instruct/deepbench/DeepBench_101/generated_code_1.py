import torch

def su2_generators(k):
    if k < 0:
        raise ValueError("The representation index 'k' must be a non-negative integer.")
    
    # Create the J_z generator matrix
    j_z = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    
    # Create the J_x and J_y generator matrices using the Pauli matrices
    j_x = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 2
    j_y = torch.tensor([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / 2
    
    # Create a tensor to hold the generators
    generators = torch.stack([j_x, j_y, j_z])
    
    return generators

if __name__ == "__main__":
    k = 1  # Example input
    generators = su2_generators(k)
    print(generators)