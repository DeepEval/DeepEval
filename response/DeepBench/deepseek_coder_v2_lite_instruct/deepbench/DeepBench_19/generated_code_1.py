import torch

def look_at_rotation(camera_position, at, up, device='cpu'):
    camera_position = torch.tensor(camera_position, device=device).float()
    at = torch.tensor(at, device=device).float()
    up = torch.tensor(up, device=device).float()

    # Normalize vectors
    z_axis = (at - camera_position).float().unsqueeze(0)
    z_axis = z_axis / torch.norm(z_axis, dim=1, keepdim=True)

    x_axis = torch.cross(up, z_axis, dim=1).float().unsqueeze(0)
    x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

    y_axis = torch.cross(z_axis, x_axis, dim=1).float().unsqueeze(0)
    y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)

    # Handle case where x-axis is close to zero
    x_axis[torch.norm(x_axis, dim=2) < 1e-8] = torch.tensor([1.0, 0.0, 0.0], device=device).float()
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)

    # Concatenate axes to form the rotation matrix
    R = torch.cat((x_axis, y_axis, z_axis), dim=0)

    return R.transpose(1, 0)

if __name__ == "__main__":
    camera_position = [1.0, 2.0, 3.0]
    at = [4.0, 5.0, 6.0]
    up = [0.0, 1.0, 0.0]
    device = 'cpu'

    R = look_at_rotation(camera_position, at, up, device)
    print(R)