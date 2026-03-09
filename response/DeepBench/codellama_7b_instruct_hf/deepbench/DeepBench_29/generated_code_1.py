import torch

def quaternion_to_axis_angle(quaternion):
    # Check if the input is a tensor and has a shape of Nx4 or 4
    if not torch.is_tensor(quaternion) or quaternion.shape != (N, 4) and quaternion.shape != (4,):
        raise ValueError("Input must be a tensor with shape (N, 4) or (4,)")

    # Unpack the quaternion components
    w, x, y, z = torch.unbind(quaternion, dim=-1)

    # Compute the axis angle of rotation
    theta = 2 * torch.atan2(torch.sqrt(w * w - x * x - y * y - z * z), x * y - w * z)
    axis = torch.cross(x, y, z)

    # Return the axis angle of rotation
    return theta, axis

if __name__ == "__main__":
    quat = torch.tensor([0.9962, 0.0872, 0.0, 0.0])
    theta, axis = quaternion_to_axis_angle(quat)
    print("Theta:", theta.item())
    print("Axis:", axis)
    quat_batch = torch.tensor([[0.9962, 0.0872, 0.0, 0.0], [0.9801, 0.0, 0.1987, 0.0]])
    theta_batch, axis_batch = quaternion_to_axis_angle(quat_batch)
    print("Batch theta:", theta_batch)
    print("Batch axis:", axis_batch)