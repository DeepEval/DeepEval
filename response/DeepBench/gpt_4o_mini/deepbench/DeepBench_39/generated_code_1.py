import torch

def SO3_generators(k):
    J_x = torch.tensor([[0, 0, 0], 
                        [0, 0, -1j], 
                        [0, 1j, 0]], dtype=torch.complex64) * (k / 2)
    
    J_y = torch.tensor([[0, 0, 1j], 
                        [0, 0, 0], 
                        [-1j, 0, 0]], dtype=torch.complex64) * (k / 2)

    J_z = torch.tensor([[0, -1j, 0], 
                        [1j, 0, 0], 
                        [0, 0, 0]], dtype=torch.complex64) * (k / 2)

    return torch.stack([J_x, J_y, J_z])

if __name__ == "__main__":
    k = 1
    generators = SO3_generators(k)
    print(generators)