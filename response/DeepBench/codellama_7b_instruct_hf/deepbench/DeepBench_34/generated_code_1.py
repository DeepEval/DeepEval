import numpy as np
from scipy.linalg import inv

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    # Check input shapes
    if E_mat.shape != (3, 3):
        raise ValueError("E_mat must be a 3x3 matrix")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3 matrices")
    if x1.shape != (N, 2) or x2.shape != (N, 2):
        raise ValueError("x1 and x2 must be Nx2 matrices")
    if mask is not None and mask.shape != (N,):
        raise ValueError("mask must be a 1D array of length N")

    # Compute essential matrix from camera matrices and point sets
    E_inv = inv(E_mat)
    K1_inv = inv(K1)
    K2_inv = inv(K2)
    x1_transformed = np.dot(K1_inv, np.dot(E_inv, x1))
    x2_transformed = np.dot(K2_inv, np.dot(E_inv, x2))

    # Compute relative camera rotation and translation
    R_mat, t_vec = compute_relative_camera_motion(x1_transformed, x2_transformed)

    # Compute 3D points from triangulated points
    points_3D = np.zeros((N, 3))
    for i in range(N):
        if mask[i] == 1:
            points_3D[i] = triangulate_point(x1_transformed[i], x2_transformed[i])

    # Return rotation matrix, translation vector, and 3D points
    return R_mat, t_vec, points_3D

def compute_relative_camera_motion(x1_transformed, x2_transformed):
    # Compute relative camera rotation and translation
    R_mat = np.zeros((3, 3))
    t_vec = np.zeros((3,))
    for i in range(N):
        for j in range(N):
            if mask[i] == 1 and mask[j] == 1:
                R_mat[i, j] = np.dot(x1_transformed[i], x2_transformed[j])
                t_vec[i] = np.dot(x1_transformed[i], x2_transformed[j])

    # Normalize rotation matrix
    R_mat /= np.linalg.norm(R_mat)

    # Return rotation matrix and translation vector
    return R_mat, t_vec

def triangulate_point(x1_transformed, x2_transformed):
    # Compute 3D point from triangulated points
    x1 = np.dot(inv(K1), np.dot(inv(E_mat), x1_transformed))
    x2 = np.dot(inv(K2), np.dot(inv(E_mat), x2_transformed))
    x3 = np.dot(inv(K1), np.dot(inv(E_mat), x1_transformed))
    x4 = np.dot(inv(K2), np.dot(inv(E_mat), x2_transformed))
    return np.dot(x1, x2, x3, x4)

if __name__ == "__main__":
    N = 10
    E_mat = np.random.randn(3, 3)
    K1 = np.eye(3)
    K2 = np.eye(3)
    x1 = np.random.randn(N, 2)
    x2 = np.random.randn(N, 2)
    mask = np.ones(N)
    R_mat, t_vec, points_3D = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask)
    print("Rotation matrix shape:", R_mat.shape)
    print("Translation vector shape:", t_vec.shape)
    print("3D points shape:", points_3D.shape)