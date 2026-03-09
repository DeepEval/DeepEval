import numpy as np
import cv2

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    if E_mat.shape[-2:] != (3, 3):
        raise ValueError("Essential matrix must be of shape (3, 3)")
    if K1.shape[-2:] != (3, 3):
        raise ValueError("Camera matrix K1 must be of shape (3, 3)")
    if K2.shape[-2:] != (3, 3):
        raise ValueError("Camera matrix K2 must be of shape (3, 3)")
    if x1.shape[-1] != 2 or x2.shape[-1] != 2:
        raise ValueError("Points must be of shape (N, 2)")

    if mask is not None:
        x1 = x1[mask]
        x2 = x2[mask]

    # Decompose the essential matrix into possible rotations and translations
    _, R1, R2, t = cv2.decomposeEssentialMat(E_mat)

    # Initialize possible rotation-translation combinations
    possible_solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]

    best_solution = None
    max_positive_depth = 0
    best_points_3d = None

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

    for R, t in possible_solutions:
        P2 = K2 @ np.hstack((R, t))

        points_4d_hom = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        # Calculate the number of points in front of both cameras
        depth1 = points_3d[2, :]
        depth2 = (R[2, :] @ points_3d) + t[2]

        num_positive_depth = np.sum((depth1 > 0) & (depth2 > 0))

        if num_positive_depth > max_positive_depth:
            max_positive_depth = num_positive_depth
            best_solution = (R, t)
            best_points_3d = points_3d.T

    if best_solution is None:
        raise ValueError("Could not find a valid solution")

    R_best, t_best = best_solution
    return R_best, t_best, best_points_3d

if __name__ == "__main__":
    # Sample input values
    E_mat = np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]], dtype=float)
    K1 = np.eye(3)
    K2 = np.eye(3)
    
    x1 = np.array([[0.5, 0.5], [0.6, 0.6], [0.4, 0.4]], dtype=float)
    x2 = np.array([[0.52, 0.48], [0.62, 0.58], [0.42, 0.38]], dtype=float)
    
    # Call the function
    R, t, points_3d = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
    
    # Print the results
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    print("3D Points:\n", points_3d)