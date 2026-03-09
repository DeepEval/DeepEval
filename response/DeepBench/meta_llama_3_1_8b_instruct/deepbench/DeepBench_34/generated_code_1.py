import numpy as np
import torch

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    # Check input shapes
    batch_size = E_mat.shape[0]
    if E_mat.shape!= (batch_size, 3, 3):
        raise ValueError("E_mat must have shape (batch_size, 3, 3)")
    if K1.shape!= (batch_size, 3, 3) or K2.shape!= (batch_size, 3, 3):
        raise ValueError("K1 and K2 must have shape (batch_size, 3, 3)")
    if x1.shape!= (batch_size, 2) or x2.shape!= (batch_size, 2):
        raise ValueError("x1 and x2 must have shape (batch_size, 2)")

    # Extract points and camera matrices
    points = np.concatenate((x1, x2), axis=1)
    K = np.concatenate((K1, K2), axis=1)

    # Check if mask is provided
    if mask is None:
        mask = np.ones((batch_size, 2), dtype=bool)

    # Decompose essential matrix
    W1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    det_E = np.linalg.det(E_mat)
    if det_E > 0:
        R1 = np.eye(3)
        R2 = W2 @ E_mat @ W1
    else:
        R1 = W2 @ E_mat @ W1
        R2 = np.eye(3)

    # Choose the best solution
    if np.linalg.det(R1) > 0:
        R = R1
    else:
        R = R2

    # Compute translation vector
    t = np.linalg.inv(K) @ R.T @ np.mean(points[mask], axis=1)

    # Triangulate 3D points
    points_3d = np.linalg.inv(R.T @ K) @ np.concatenate((points[mask], np.ones((batch_size, 1))), axis=1).T

    return R, t, points_3d

if __name__ == "__main__":
    # Create sample input values
    batch_size = 1
    E_mat = np.random.rand(batch_size, 3, 3)
    K1 = np.random.rand(batch_size, 3, 3)
    K2 = np.random.rand(batch_size, 3, 3)
    x1 = np.random.rand(batch_size, 2)
    x2 = np.random.rand(batch_size, 2)

    # Call the function
    R, t, points_3d = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)

    # Print the results
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector:")
    print(t)
    print("Triangulated 3D Points:")
    print(points_3d)