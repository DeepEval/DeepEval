import torch

def look_at_rotation(camera_position, at, up, device='cpu'):
    def normalize(vectors):
        return vectors / torch.norm(vectors, dim=-1, keepdim=True)
    
    if isinstance(camera_position, (tuple, list)):
        camera_position = torch.tensor(camera_position, dtype=torch.float32, device=device)
    if isinstance(at, (tuple, list)):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    if isinstance(up, (tuple, list)):
        up = torch.tensor(up, dtype=torch.float32, device=device)

    camera_position = camera_position.to(device)
    at = at.to(device)
    up = up.to(device)

    z_axis = normalize(camera_position - at)
    x_axis = normalize(torch.cross(up, z_axis))
    
    # Handle case where x_axis is close to zero
    x_axis = torch.where(torch.norm(x_axis, dim=-1, keepdim=True) < 1e-6,
                         torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32).expand_as(x_axis),
                         x_axis)
    
    y_axis = torch.cross(z_axis, x_axis)
    
    # Concatenate axes to form the rotation matrix
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    
    return R.transpose(-2, -1)

if __name__ == "__main__":
    camera_position = [(0.0, 0.0, 1.0)]
    at = [(0.0, 0.0, 0.0)]
    up = [(0.0, 1.0, 0.0)]

    R = look_at_rotation(camera_position, at, up, device='cpu')
    print("Rotation Matrix:")
    print(R)