import torch
import torch.nn as nn
import numpy as np

def su2_generators(k):
    J_x = torch.zeros((2*k + 1, 2*k + 1), requires_grad=False)
    J_y = torch.zeros((2*k + 1, 2*k + 1), requires_grad=False)
    J_z = torch.zeros((2*k + 1, 2*k + 1), requires_grad=False)
    
    for i in range(2*k + 1):
        J_x[i, i] = 0.5 * torch.tensor([[0, -1], [1, 0]], dtype=torch.complex128)
        J_y[i, i] = 0.5j * torch.tensor([[0, -1], [1, 0]], dtype=torch.complex128)
        J_z[i, i] = torch.tensor([[i, 0], [0, -i]], dtype=torch.complex128)
    
    J_x = J_x / (2*k)
    J_y = J_y / (2*k)
    J_z = J_z / (2*k)
    
    J = torch.stack([J_x, J_y, J_z], dim=0)
    
    return J

if __name__ == "__main__":
    k = 2
    J = su2_generators(k)
    print(f"J_x: {J[0]}\nJ_y: {J[1]}\nJ_z: {J[2]}")