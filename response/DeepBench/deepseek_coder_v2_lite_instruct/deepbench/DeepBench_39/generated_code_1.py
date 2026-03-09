import torch

def so3_generators(k):
    # Create the generators J_x, J_y, J_z in the SO(3) basis
    J_x = torch.tensor([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    J_y = torch.tensor([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
    J_z = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    
    # Stack the generators into a single tensor
    generators = torch.stack([J_x, J_y, J_z])
    
    return generators

if __name__ == "__main__":
    # Create sample input values
    k = 1
    
    # Call the function
    generators = so3_generators(k)
    
    # Print the results
    print(generators)