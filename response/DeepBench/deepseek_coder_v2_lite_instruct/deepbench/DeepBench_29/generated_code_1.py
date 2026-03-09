import torch
import math

def quaternion_to_axis_angle(quaternion):
    # Check if the input is a tensor and has the required shape
    if not isinstance(quaternion, torch.Tensor) or quaternion.shape[-1] != 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4")
    
    # Unpack the quaternion components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # Compute the magnitude of the quaternion
    norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
    
    # Normalize the quaternion
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    
    # Compute the angle of rotation
    angle = 2 * torch.arccos(w)
    
    # Compute the axis of rotation
    axis = torch.stack([x, y, z], dim=-1)
    
    # Avoid division by zero
    safe_norm = torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis / safe_norm
    
    # Convert the angle to radians
    axis_angle = angle * axis
    
    return axis_angle

if __name__ == "__main__":
    # Example usage
    quaternion = torch.tensor([0.7071, 0.0000, 0.0000, 0.7071])  # Example quaternion (90 degrees around z-axis)
    axis_angle = quaternion_to_axis_angle(quaternion)
    print(axis_angle)  # Expected output: tensor([0.0000, 0.0000, 0.7854])