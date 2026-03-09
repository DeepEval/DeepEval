import numpy as np

def quaternion_to_rotation_matrix(quaternion):
    # Normalize the quaternion
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Quaternion must not be zero")
    normalized_quaternion = quaternion / norm
    
    # Unpack the quaternion components
    w, x, y, z = normalized_quaternion
    
    # Compute the rotation matrix
    q11 = 2 * (w**2 + x**2) - 1
    q12 = 2 * (w*x - y*z)
    q13 = 2 * (w*y + z*x)
    
    q21 = 2 * (w*x + y*z)
    q22 = 2 * (y**2 + z**2) - 1
    q23 = 2 * (y*z - w*x)
    
    q31 = 2 * (w*y - z*x)
    q32 = 2 * (w*z + x*y)
    q33 = 2 * (x**2 + z**2) - 1
    
    # Reshape the output matrix
    rotation_matrix = np.array([[q11, q12, q13],
                                [q21, q22, q23],
                                [q31, q32, q33]])
    
    return rotation_matrix.reshape(*quaternion.shape[:-1], 3, 3)

if __name__ == "__main__":
    # Example usage
    import random
    quaternion = np.random.rand(3, 4)  # Random quaternion
    print("Quaternion:", quaternion)
    
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    print("Rotation Matrix:\n", rotation_matrix)