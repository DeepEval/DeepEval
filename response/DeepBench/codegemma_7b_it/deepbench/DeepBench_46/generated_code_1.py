import torch

def quaternion_exp_to_log(quaternion, eps):
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a Tensor")
    if quaternion.shape[-1] != 4:
        raise ValueError("Input must have a shape of (*, 4)")

    v, w = quaternion[..., 1:], quaternion[..., 0]
    v_norm = v.norm(dim=-1, keepdim=True)
    w_norm = (1 + w).clamp_min(eps).sqrt()
    qw_inv = w / w_norm
    qv_inv = v / (w_norm * v_norm)

    log_factor = torch.max(
        torch.log(w_norm),
        torch.log(v_norm + (qw_inv * v).sum(dim=-1, keepdim=True))
    )

    output = torch.cat([log_factor * qv_inv, log_factor], dim=-1)

    return output

if __name__ == "__main__":
    # Sample input values
    quaternion = torch.tensor([[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5]])
    eps = 1e-5

    # Call the function
    quaternion_log = quaternion_exp_to_log(quaternion, eps)

    # Print the results
    print("Input quaternion:", quaternion)
    print("Quaternion log map:", quaternion_log)