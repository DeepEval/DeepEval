import torch

def look_at_rotation(camera_position, at, up, device):
    camera_position = torch.as_tensor(camera_position, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)
    at = torch.as_tensor(at, dtype=torch.float32, device=device)

    if camera_position.ndim == 1:
        camera_position = camera_position.unsqueeze(0)
    if at.ndim == 1:
        at = at.unsqueeze(0)
    if up.ndim == 1:
        up = up.unsqueeze(0)

    forward = (at - camera_position).normalize(dim=-1)
    side = up.cross(forward).normalize(dim=-1)
    up = forward.cross(side).normalize(dim=-1)

    if (side < 1e-6).all():
        side = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    R00 = side[0]
    R01 = side[1]
    R02 = side[2]
    R10 = up[0]
    R11 = up[1]
    R12 = up[2]
    R20 = forward[0]
    R21 = forward[1]
    R22 = forward[2]

    R = torch.stack([
        [R00, R01, R02],
        [R10, R11, R12],
        [R20, R21, R22],
    ], dim=-1).to(device)

    return R.transpose(-2, -1)

if __name__ == "__main__":
    camera_position = [0.0, 0.0, 5.0]
    at = [0.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]
    device = torch.device("cpu")

    R = look_at_rotation(camera_position, at, up, device)

    print(R)