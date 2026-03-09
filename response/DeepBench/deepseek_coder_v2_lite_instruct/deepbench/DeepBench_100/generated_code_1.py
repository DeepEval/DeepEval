import torch

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=-1)
    
    # Unpack the quaternion components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # Compute the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Construct the rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)
    
    return R

if __name__ == "__main__":
    # Create sample input values
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])
    
    # Call the function
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    
    # Print the results
    print("Quaternion:")
    print(quaternion)
    print("Rotation Matrix:")
    print(rotation_matrix)