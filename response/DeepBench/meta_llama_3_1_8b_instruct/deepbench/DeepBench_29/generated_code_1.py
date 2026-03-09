import numpy as np
import torch

def quaternion_to_axis_angle(quaternion):
    if not isinstance(quaternion, torch.Tensor):
        raise ValueError("Input must be a tensor")
    if quaternion.shape[-1] not in [4]:
        raise ValueError("Input must have a shape of Nx4 or 4")
    
    quaternion = quaternion.contiguous()
    w, x, y, z = quaternion.split(1, dim=-1)
    
    axis = torch.sqrt(x.pow(2) + y.pow(2) + z.pow(2))
    angle = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.pow(2) + y.pow(2)))
    
    return torch.cat([axis, angle], dim=-1)

if __name__ == "__main__":
    # Create a sample quaternion tensor
    quaternion = torch.tensor([0.5, 0.3, 0.2, 0.1], dtype=torch.float32)
    
    # Reshape to Nx4 tensor
    quaternion = quaternion.reshape(-1, 4)
    
    # Create multiple quaternions for Nx4 shape
    quaternion = torch.cat([quaternion, quaternion], dim=0)
    
    # Call the function
    result = quaternion_to_axis_angle(quaternion)
    
    # Print the results
    print(result)