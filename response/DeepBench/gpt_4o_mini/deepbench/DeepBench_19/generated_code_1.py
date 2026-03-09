import torch

def look_at_rotation(camera_position, at, up, device='cpu'):
    camera_position = torch.tensor(camera_position, dtype=torch.float32, device=device)
    at = torch.tensor(at, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)

    # Normalize the vectors
    z_axis = (camera_position - at)
    z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)

    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True)

    # Recompute the y_axis
    y_axis = torch.cross(z_axis, x_axis)

    # Handle the case where x_axis is close to zero
    if torch.any(x_axis.norm(dim=-1) < 1e-6):
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True)
        x_axis = torch.cross(y_axis, z_axis)

    R = torch.stack((x_axis, y_axis, -z_axis), dim=-1)
    return R.transpose(-1, -2)

if __name__ == "__main__":
    camera_position = [[0, 0, 1]]
    at = [[0, 0, 0]]
    up = [[0, 1, 0]]
    rotation_matrix = look_at_rotation(camera_position, at, up)
    print(rotation_matrix)