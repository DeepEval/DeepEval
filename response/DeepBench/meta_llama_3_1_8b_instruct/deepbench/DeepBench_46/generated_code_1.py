import numpy as np
import torch

def quaternion_exp_to_log(quaternion, eps):
    if not isinstance(quaternion, torch.Tensor) or quaternion.shape[-1]!= 4:
        raise ValueError("Input must be a Tensor of shape (*, 4)")
    
    w, x, y, z = quaternion.unbind(-1)
    quaternion_norm = torch.sqrt(w * w + x * x + y * y + z * z + eps)
    w, x, y, z = w / quaternion_norm, x / quaternion_norm, y / quaternion_norm, z / quaternion_norm
    
    quaternion_log = torch.stack([x, y, z], dim=-1)
    
    return quaternion_log

if __name__ == "__main__":
    # Create sample input values
    quaternion = torch.tensor([0.5, 0.3, 0.2, 0.1])
    eps = 1e-6
    
    # Call the function
    result = quaternion_exp_to_log(quaternion.unsqueeze(0), eps)
    
    # Print the results
    print(result)