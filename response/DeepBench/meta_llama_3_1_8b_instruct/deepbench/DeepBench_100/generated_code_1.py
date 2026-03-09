import numpy as np

def quaternion_to_rotation_matrix(quaternion):
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    rotation_matrix = np.stack([
        (1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w),
        (2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w),
        (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2)
    ], axis=-1)
    return rotation_matrix

if __name__ == "__main__":
    # Create sample quaternion tensor
    quaternion = np.random.rand(2, 4)
    print("Quaternion:")
    print(quaternion)

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    print("\nRotation Matrix:")
    print(rotation_matrix)