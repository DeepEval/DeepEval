import numpy as np
from scipy.linalg import null_space

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    # Check input shapes
    if E_mat.shape != (3, 3):
        raise ValueError("Essential matrix must be of shape (3, 3)")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("Camera matrices must be of shape (3, 3)")
    if x1.shape != x2.shape or x1.shape[0] < 4 or x1.shape[1] != 2:
        raise ValueError("Point sets must be of shape (num_points, 2)")
    
    # Check mask shape
    if mask is not None and (mask.shape != x1.shape[:-1] or mask.dtype != bool):
        raise ValueError("Mask must be of shape (num_points,) and of type bool")
    
    # Normalize points
    x1_norm = np.linalg.inv(K1) @ np.vstack((x1.T, np.ones(x1.shape[0])))
    x2_norm = np.linalg.inv(K2) @ np.vstack((x2.T, np.ones(x2.shape[0])))
    
    # Compute all possible solutions for rotation and translation
    U, S, Vt = np.linalg.svd(E_mat)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    
    # Possible rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    R3 = U @ Z @ W @ Vt
    R4 = U @ Z.T @ W @ Vt
    
    # Possible translation vectors
    t1 = U[:, 2]
    t2 = -U[:, 2]
    t3 = U[:, 2]
    t4 = -U[:, 2]
    
    # Solutions
    solutions = [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2),
        (R3, t3),
        (R3, t4),
        (R4, t3),
        (R4, t4)
    ]
    
    # Score solutions based on triangulation
    best_score = -1
    best_R = None
    best_t = None
    best_points = None
    
    for R, t in solutions:
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t.reshape(3, 1)))
        
        # Triangulate points
        X = np.zeros((4, x1.shape[0]))
        for i in range(x1.shape[0]):
            if mask is None or mask[i]:
                A = np.array([
                    x1_norm[0, i] * P1[2, :] - P1[0, :],
                    x1_norm[1, i] * P1[2, :] - P1[1, :],
                    x2_norm[0, i] * P2[2, :] - P2[0, :],
                    x2_norm[1, i] * P2[2, :] - P2[1, :]
                ])
                X[:, i] = null_space(A)[:4]
        
        # Score based on triangulation
        score = np.sum(X[2, :] > 0)
        
        # Update best solution
        if score > best_score:
            best_score = score
            best_R = R
            best_t = t
            best_points = X[:3, :] / X[3, :]
    
    return best_R, best_t, best_points

if __name__ == "__main__":
    # Sample input values
    E_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    K1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    x2 = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
    
    # Call the function and print the results
    R, t, points = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    print("3D Points:\n", points)