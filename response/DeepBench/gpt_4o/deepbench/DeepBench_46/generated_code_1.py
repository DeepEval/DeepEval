import torch

def quaternion_exp_to_log(quaternion, eps=1e-6):
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a PyTorch Tensor")
    
    if quaternion.shape[-1] != 4:
        raise ValueError("Input tensor must have a shape of (*, 4)")

    # Unpack quaternion into scalar and vector parts
    q_w = quaternion[..., 0]
    q_vec = quaternion[..., 1:]

    # Compute the norm of the vector part
    vec_norm = torch.norm(q_vec, dim=-1, keepdim=True)

    # Clamp the norm to avoid division by zero
    vec_norm = vec_norm.clamp(min=eps)

    # Compute the angle theta
    theta = torch.atan2(vec_norm, q_w.unsqueeze(-1))

    # Normalize the vector part
    log_map = q_vec * (theta / vec_norm)

    return log_map.squeeze(-1)

if __name__ == "__main__":
    # Create a sample quaternion input
    sample_quaternion = torch.tensor([[0.9239, 0.3827, 0.0, 0.0],
                                      [1.0, 0.0, 0.0, 0.0]])

    # Call the function
    log_map_result = quaternion_exp_to_log(sample_quaternion)

    # Print the results
    print("Quaternion Log Map:")
    print(log_map_result)