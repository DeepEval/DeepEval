import torch

def quaternion_to_rotation_matrix(quaternion):
    normalized_quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    w, x, y, z = normalized_quaternion.chunk(4, dim=-1)
    return torch.stack((
        torch.stack((1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w), dim=-1),
        torch.stack((2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w), dim=-1),
        torch.stack((2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2), dim=-1)
    ), dim=-2)

if __name__ == "__main__":
    sample_quaternion = torch.randn(3, 4)
    rotation_matrix = quaternion_to_rotation_matrix(sample_quaternion)
    print(rotation_matrix)