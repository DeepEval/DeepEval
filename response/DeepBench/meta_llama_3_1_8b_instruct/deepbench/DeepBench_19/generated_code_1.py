import torch
import numpy as np

def look_at_rotation(camera_position, at, up, device):
    # Normalize vectors
    camera_position = camera_position / torch.norm(camera_position, dim=-1, keepdim=True)
    at = at / torch.norm(at, dim=-1, keepdim=True)
    up = up / torch.norm(up, dim=-1, keepdim=True)

    # Calculate z-axis
    z_axis = (camera_position - at).detach().requires_grad_(True)

    # Calculate x-axis
    x_axis = torch.cross(up, z_axis, dim=-1)
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

    # Handle case where x-axis is close to zero
    x_axis = torch.where(x_axis.abs().max(dim=-1, keepdim=True)[0] < 1e-6,
                         up, x_axis)

    # Calculate y-axis
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)

    # Concatenate axes to form rotation matrix R
    R = torch.cat([x_axis, y_axis, z_axis], dim=-1)
    R = R.transpose(-2, -1)

    return R

if __name__ == "__main__":
    # Create sample input values
    camera_position = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    at = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    up = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    device = torch.device("cpu")

    # Call the function and print the results
    R = look_at_rotation(camera_position, at, up, device)
    print(R)