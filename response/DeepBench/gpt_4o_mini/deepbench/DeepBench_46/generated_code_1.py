import torch

def quaternion_exp_to_log(quaternion, eps=1e-6):
    if not isinstance(quaternion, torch.Tensor) or quaternion.shape[-1] != 4:
        raise ValueError("Input must be a tensor of shape (*, 4)")
    
    w, x, y, z = quaternion.unbind(dim=-1)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    
    # Clamp the norm to avoid division by zero
    norm = torch.clamp(norm, min=eps)
    
    theta = torch.atan2(norm, w)
    sin_theta = torch.sin(theta)
    
    log_map = (theta / sin_theta).unsqueeze(-1) * torch.stack((x, y, z), dim=-1)
    
    return log_map

if __name__ == "__main__":
    # Create a sample quaternion tensor
    sample_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                       [0.7071, 0.7071, 0.0, 0.0],
                                       [0.0, 0.0, 0.7071, 0.7071]])
    
    # Call the function and print the results
    result = quaternion_exp_to_log(sample_quaternion)
    print(result)