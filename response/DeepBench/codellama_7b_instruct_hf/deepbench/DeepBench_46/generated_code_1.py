import torch

def quaternion_exp_to_log(quaternion, eps=1e-8):
    """
    Applies the log map to a quaternion.

    Args:
        quaternion (Tensor): The quaternion in (w, x, y, z) format.
        eps (float, optional): The clamping value for the quaternion norm.

    Returns:
        Tensor: The quaternion log map.
    """
    if not torch.is_tensor(quaternion) or quaternion.shape[-1] != 4:
        raise ValueError("The input must be a Tensor of shape (*, 4).")

    w, x, y, z = quaternion.unbind()
    norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
    if norm < eps:
        raise ValueError("The quaternion norm must be greater than eps.")

    log_w = torch.log(norm)
    log_x = torch.log(x)
    log_y = torch.log(y)
    log_z = torch.log(z)

    return torch.stack((log_w, log_x, log_y, log_z), dim=-1)

if __name__ == "__main__":
    # Test the function with a sample input
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    log_quaternion = quaternion_exp_to_log(quaternion)
    print(log_quaternion)