import torch

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    
    # Unpack the quaternion components
    w, x, y, z = quaternion.unbind(dim=-1)
    
    # Compute the rotation matrix components
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Construct the rotation matrix
    rotation_matrix = torch.stack((
        ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz
    ), dim=-1).reshape(-1, 3, 3)
    
    return rotation_matrix

if __name__ == "__main__":
    # Example usage
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    print(rotation_matrices)