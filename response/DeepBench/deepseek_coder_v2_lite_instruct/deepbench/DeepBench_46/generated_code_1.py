import torch

def quaternion_exp_to_log(quaternion, eps):
    if not isinstance(quaternion, torch.Tensor) or quaternion.shape[-1] != 4:
        raise ValueError("Input must be a Tensor of shape (*, 4)")
    
    w, v = quaternion[..., 0], quaternion[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=eps)
    
    theta = torch.atan2(v_norm, w)
    theta_over_v_norm = torch.where(theta == 0, torch.tensor(1.0, device=theta.device), theta / v_norm)
    
    log_map = torch.where(v_norm == 0, torch.zeros_like(quaternion[..., :3]), theta_over_v_norm * v)
    
    return log_map

if __name__ == "__main__":
    # Example usage
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
    eps = 1e-8
    result = quaternion_exp_to_log(quaternion, eps)
    print(result)