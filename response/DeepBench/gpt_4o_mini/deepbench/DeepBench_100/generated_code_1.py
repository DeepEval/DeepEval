import torch

def quaternion_to_rotation_matrix(quaternion):
    norm_quat = torch.norm(quaternion, dim=-1, keepdim=True)
    normalized_quat = quaternion / norm_quat

    w, x, y, z = normalized_quat.unbind(dim=-1)

    rotation_matrix = torch.empty(*quaternion.shape[:-1], 3, 3, device=quaternion.device)
    rotation_matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[..., 0, 1] = 2 * (x * y - z * w)
    rotation_matrix[..., 0, 2] = 2 * (x * z + y * w)

    rotation_matrix[..., 1, 0] = 2 * (x * y + z * w)
    rotation_matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[..., 1, 2] = 2 * (y * z - x * w)

    rotation_matrix[..., 2, 0] = 2 * (x * z - y * w)
    rotation_matrix[..., 2, 1] = 2 * (y * z + x * w)
    rotation_matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrix

if __name__ == "__main__":
    sample_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], 
                                       [0.7071, 0.7071, 0.0, 0.0], 
                                       [0.0, 0.7071, 0.7071, 0.0]])
    rotation_matrices = quaternion_to_rotation_matrix(sample_quaternion)
    print(rotation_matrices)