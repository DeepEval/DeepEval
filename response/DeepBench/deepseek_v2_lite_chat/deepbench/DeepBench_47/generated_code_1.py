import numpy as np
from scipy.linalg import svd
import itertools

def normalize_points(points):
    """Normalize 2D points to unit circle."""
    x, y = points[:, :, 0], points[:, :, 1]
    norm = np.sqrt(x**2 + y**2)
    points_normalized = points / norm[:, :, np.newaxis]
    return points_normalized

def compute_fundamental_matrix(p1, p2):
    """Compute the fundamental matrix using the 7-point algorithm."""
    assert p1.shape == p2.shape == (5, 8), "Input points do not match the required shape"
    
    # Normalize points
    p1_norm = normalize_points(p1)
    p2_norm = normalize_points(p2)
    
    # Compute the A and b matrices
    A = np.zeros((9, 9))
    b = np.zeros((9, 1))
    
    for I in range(9):
        for J in range(9):
            if I == J:
                A[I, J] = -(p1_norm[0, :] @ p1_norm[0, :]).sum()
            else:
                A[I, J] = p1_norm[0, :] @ p1_norm[1, :]
                A[I, J] -= p2_norm[0, :] @ p2_norm[1, :]
    
    A[8, :] = -np.sum(p1_norm[0, :] * p1_norm[2, :])
    A[8, :] -= np.sum(p2_norm[0, :] * p2_norm[2, :])
    
    b[8, :] = -(p1_norm[0, :] * p1_norm[3, :]).sum()
    b[8, :] -= -(p2_norm[0, :] * p2_norm[3, :]).sum()
    
    # Solve the linear system using SVD
    U, S, Vt = svd(A)
    s = S[np.isclose(S, 0)]  # Remove zero singular values
    
    # Handle non-invertible matrices
    if len(s) < 6:
        raise AssertionError("Matrix is non-invertible")
    
    F = Vt.T[-1, :][:, np.newaxis]
    F /= F[2, 0]
    
    return F

def run_7point(points1, points2):
    """Compute up to three potential fundamental matrices."""
    assert points1.shape == points2.shape == (5, 8), "Input points do not match the required shape"
    
    F1 = compute_fundamental_matrix(points1, points2)
    F2 = compute_fundamental_matrix(points1, points2 + np.array([[0, 0, 0, 1, 0]]))  # Shift + scaling
    F3 = compute_fundamental_matrix(points1 + np.array([[0, 0, 100, 0, 0]]), points2)  # Translation
    
    return np.array([F1, F2, F3])

if __name__ == "__main__":
    # Create sample input values
    batch_size = 1
    num_points = 7
    points1 = np.random.rand(batch_size, num_points, 2)
    points2 = np.random.rand(batch_size, num_points, 2)
    
    # Call the function and print the results
    F = run_7point(points1, points2)
    print("Fundamental matrices:\n", F)