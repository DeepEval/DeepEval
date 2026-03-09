import torch
import torch.nn.functional as F

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    norm = w**2 + x**2 + y**2 + z**2
    norm = torch.sqrt(norm)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    qw, qx, qy, qz = quaternion
    R = torch.zeros(quaternion.shape[0], 3, 3)
    R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    R[:, 0, 1] = 2*x*y - 2*qw*qz
    R[:, 0, 2] = 2*x*z + 2*qw*qy
    R[:, 1, 0] = 2*x*y + 2*qw*qz
    R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    R[:, 1, 2] = 2*y*z - 2*qw*qx
    R[:, 2, 0] = 2*x*z - 2*qw*qy
    R[:, 2, 1] = 2*y*z + 2*qw*qx
    R[:, 2, 2] = 1 - 2*x**2 - 2*y**2
    return R

if __name__ == "__main__":
    quaternion = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = quaternion_to_rotation_matrix(quaternion)
    print(R)