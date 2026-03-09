import numpy as np
import cv2

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    if E_mat.shape != (3, 3):
        raise ValueError("Essential matrix E_mat must have shape (3, 3)")
    if K1.shape[0] != K1.shape[1] or K2.shape[0] != K2.shape[1] or K1.shape[0] != 3 or K2.shape[0] != 3:
        raise ValueError("Camera matrices K1 and K2 must be square matrices of shape (3, 3)")
    if x1.shape[1] != 2 or x2.shape[1] != 2 or x1.shape[0] != x2.shape[0]:
        raise ValueError("Point sets x1 and x2 must have the same number of points with shape (N, 2)")
    
    if mask is not None:
        x1 = x1[mask.flatten()]
        x2 = x2[mask.flatten()]

    points1 = cv2.undistortPoints(np.expand_dims(x1, axis=1), K1, None)
    points2 = cv2.undistortPoints(np.expand_dims(x2, axis=1), K2, None)

    retval, R, t, _ = cv2.recoverPose(E_mat, points1, points2)

    points_4d_hom = cv2.triangulatePoints(K1 @ np.hstack((np.eye(3), np.zeros((3, 1)))), K2 @ np.hstack((R, t)), x1.T, x2.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    return R, t, points_3d.T

if __name__ == "__main__":
    E_mat = np.array([[0, -1, 0], [1, 0, -0.5], [0.5, 0, 0]])
    K1 = np.eye(3)
    K2 = np.eye(3)
    x1 = np.array([[0, 0], [1, 1], [2, 2]])
    x2 = np.array([[0, 0], [1, 0], [2, 1]])
    
    R, t, points_3d = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    print("3D Points:\n", points_3d)