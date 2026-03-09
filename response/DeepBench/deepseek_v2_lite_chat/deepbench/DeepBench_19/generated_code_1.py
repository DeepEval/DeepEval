import torch
import numpy as np

def look_at_rotation(camera_position, at, up, device='cpu'):
    # Normalize vectors
    camera_position = camera_position / torch.linalg.norm(camera_position, dim=-1, keepdim=True)
    at = at / torch.linalg.norm(at - camera_position, dim=-1, keepdim=True)
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)
    
    # Cross product to get the x, y, z axes
    x = torch.cross(up, at)
    y = torch.cross(at, x)
    x, y, z = x.unbind(dim=-1)
    
    # Handle cases where x-axis is close to zero
    small_x = torch.abs(x) < 1e-6
    x = x.masked_fill(small_x, 0)
    
    # Rotation matrix
    R = torch.stack([torch.stack([x, y, z]), torch.zeros(3, device=device)], dim=1)
    
    # Transpose the matrix to get the expected output shape
    return torch.transpose(R, 0, 1)

if __name__ == "__main__":
    # Create sample input values
    camera_position = torch.tensor([0.0, 0.0, 0.0]).to(device='cpu')
    at = torch.tensor([0.0, 0.0, -1.0]).to(device='cpu')
    up = torch.tensor([0.0, 1.0, 0.0]).to(device='cpu')
    
    # Call the function and print the results
    R = look_at_rotation(camera_position, at, up)
    print("Rotation Matrix:")
    print(R)