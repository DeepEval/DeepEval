import torch

def quaternion_to_axis_angle(quaternion):
    if not isinstance(quaternion, torch.Tensor):
        raise ValueError("Input must be a tensor.")
    if quaternion.shape not in {(4,), (None, 4)}:
        raise ValueError("Input tensor must have a shape of (4,) or (N, 4).")
    
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    angle = 2 * torch.acos(w)
    s = torch.sqrt(1 - w * w)

    axis = torch.zeros_like(quaternion)
    axis[..., 0] = x / s
    axis[..., 1] = y / s
    axis[..., 2] = z / s
    
    axis[torch.isnan(axis)] = 0  # Handle the case where s == 0 (no rotation)
    
    return torch.cat((axis, angle.unsqueeze(-1)), dim=-1)

if __name__ == "__main__":
    # Sample input: quaternion representing a 90-degree rotation around the z-axis
    sample_quaternion = torch.tensor([[0.7071, 0, 0, 0.7071], [1, 0, 0, 0]])  # Two quaternions
    result = quaternion_to_axis_angle(sample_quaternion)
    print(result)