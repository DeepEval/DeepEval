import torch

def quaternion_to_rotation_matrix(quaternion):
    norm = torch.norm(quaternion, dim=1, keepdim=True)
    normalized_quaternion = quaternion / norm
    w, x, y, z = torch.unbind(normalized_quaternion, dim=1)

    Rxx = 1 - 2 * (y * y + z * z)
    Rxy = 2 * (x * y - w * z)
    Rxz = 2 * (x * z + w * y)

    Ryx = 2 * (x * y + w * z)
    Ryy = 1 - 2 * (x * x + z * z)
    Ryz = 2 * (y * z - w * x)

    Rzx = 2 * (x * z - w * y)
    Rzy = 2 * (y * z + w * x)
    Rzz = 1 - 2 * (x * x + y * y)

    rotation_matrix = torch.stack([
        torch.stack([Rxx, Rxy, Rxz]),
        torch.stack([Ryx, Ryy, Ryz]),
        torch.stack([Rzx, Rzy, Rzz]),
    ], dim=1)

    return rotation_matrix

if __name__ == "__main__":
    quaternion = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    print(rotation_matrix)