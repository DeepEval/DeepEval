import numpy as np
from scipy.linalg import svd

def run_7point(points1, points2):
    # Assert input shapes are correct
    assert points1.shape == (B, N, 2)
    assert points2.shape == (B, N, 2)

    # Normalize points
    points1_normalized = points1 / np.linalg.norm(points1, axis=2, keepdims=True)
    points2_normalized = points2 / np.linalg.norm(points2, axis=2, keepdims=True)

    # Construct linear system
    A = np.zeros((B, 9))
    for i in range(B):
        for j in range(N):
            A[i, 3*j] = points1_normalized[i, j, 0]
            A[i, 3*j + 1] = points1_normalized[i, j, 1]
            A[i, 3*j + 2] = 1

            A[i, 3*j + 3] = points2_normalized[i, j, 0]
            A[i, 3*j + 4] = points2_normalized[i, j, 1]
            A[i, 3*j + 5] = 1

            A[i, 3*j + 6] = points2_normalized[i, j, 0]
            A[i, 3*j + 7] = points2_normalized[i, j, 1]
            A[i, 3*j + 8] = -1

    # Solve linear system using SVD
    U, s, Vt = svd(A)
    W = U @ np.diag(s) @ Vt

    # Compute cubic polynomial and solve for its roots
    m = W.shape[1] // 3
    cubic_poly = np.zeros((B, 3*m, 3))
    for i in range(B):
        for j in range(m):
            cubic_poly[i, 3*j] = W[i, 3*j + 0]
            cubic_poly[i, 3*j + 1] = W[i, 3*j + 1]
            cubic_poly[i, 3*j + 2] = W[i, 3*j + 2]

    roots = np.roots(cubic_poly)

    # Compute potential fundamental matrices
    fundamental_matrices = np.zeros((B, 3*m, 3))
    for i in range(B):
        for j in range(m):
            fundamental_matrices[i, 3*j] = np.array([[1, 0, roots[i, 0]],
                                                     [0, 1, 0],
                                                     [0, 0, 1]])
            fundamental_matrices[i, 3*j + 1] = np.array([[1, roots[i, 1], 0],
                                                         [0, 1, 0],
                                                         [0, 0, 1]])
            fundamental_matrices[i, 3*j + 2] = np.array([[1, 0, 0],
                                                         [0, 1, roots[i, 2]],
                                                         [0, 0, 1]])

    # Handle edge cases
    for i in range(B):
        # Check if any of the fundamental matrices are non-invertible
        if np.linalg.det(fundamental_matrices[i]) == 0:
            # Replace non-invertible matrices with the identity matrix
            fundamental_matrices[i] = np.eye(3)

    return fundamental_matrices

if __name__ == "__main__":
    B = 2
    N = 7
    points1 = np.random.randn(B, N, 2)
    points2 = np.random.randn(B, N, 2)
    fundamental_matrices = run_7point(points1, points2)
    print("Fundamental matrices shape:", fundamental_matrices.shape)