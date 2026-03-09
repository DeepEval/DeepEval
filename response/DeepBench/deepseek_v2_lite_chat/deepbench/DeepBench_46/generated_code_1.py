import torch

def quaternion_exp_to_log(quaternion, eps):
    # Unpack quaternion into vector and scalar
    w, x, y, z = torch.unbind(quaternion, dim=-2)
    
    # Compute the quaternion norm
    norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
    
    # Clamp if norm is too large
    if norm > eps:
        normalized = quaternion / norm.unsqueeze(-1)
    else:
        # If norm is 0, we need to handle this case separately
        if torch.all(quaternion == 0):
            return torch.zeros_like(quaternion)
        else:
            raise ValueError("The quaternion norm is too large.")
    
    # Compute the logarithm of the quaternion
    log_norm = torch.log(eps + norm)
    log_x, log_y, log_z = torch.unbind(log_norm, dim=-1)
    
    # Compute the logarithm of the vector part
    log_vector = torch.stack([log_x * normalized[..., 0], log_y * normalized[..., 1], log_z * normalized[..., 2]], dim=-1)
    
    # Sum the vector part with the scalar part
    quaternion_log = torch.cat((log_vector, log_norm), dim=-1)
    
    return quaternion_log

if __name__ == "__main__":
    # Sample input values
    quaternion = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    eps = 1e-6  # Small positive value for clamping

    # Call the function and print the results
    result = quaternion_exp_to_log(quaternion, eps)
    print("Quaternion Log Map:", result)