import torch
import math

def quaternion_to_axis_angle(quaternion):
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if quaternion.shape[-1] != 4:
        raise ValueError("Input tensor must have a shape of Nx4 or 4")

    # Ensure the tensor is at least 2D for consistent indexing
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # Normalize the quaternion if it's not already a unit quaternion
    norm_quaternion = torch.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm_quaternion, x / norm_quaternion, y / norm_quaternion, z / norm_quaternion

    # Compute the angle
    angle = 2 * torch.acos(w)
    
    # Compute the axis
    s = torch.sqrt(1 - w**2)
    small_s = torch.isclose(s, torch.tensor(0.0))
    
    # Avoid division by zero by setting axis to any valid unit vector
    x = torch.where(small_s, torch.tensor(1.0), x / s)
    y = torch.where(small_s, torch.tensor(0.0), y / s)
    z = torch.where(small_s, torch.tensor(0.0), z / s)
    
    axis_angle = torch.stack([angle, x, y, z], dim=-1)
    
    return axis_angle.squeeze(0) if quaternion.shape[0] == 1 else axis_angle

if __name__ == "__main__":
    # Example usage
    quaternion_input = torch.tensor([0.8660254, 0.0, 0.5, 0.0])  # Quaternion representing a 60 degree rotation around the y-axis
    axis_angle_output = quaternion_to_axis_angle(quaternion_input)
    print("Axis-Angle Output:", axis_angle_output)
    
    # Example with batch of quaternions
    quaternions_input = torch.tensor([
        [0.8660254, 0.0, 0.5, 0.0],
        [0.7071068, 0.0, 0.7071068, 0.0]
    ])
    axis_angles_output = quaternion_to_axis_angle(quaternions_input)
    print("Batch Axis-Angle Output:", axis_angles_output)