import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    """
    Recover relative camera rotation and translation from estimated essential matrix.

    Args:
      E_mat: Estimated essential matrix (B x 3 x 3).
      K1: Camera matrix of the first image (B x 3 x 3).
      K2: Camera matrix of the second image (B x 3 x 3).
      x1: Coordinates of points in the first image (B x N x 2).
      x2: Coordinates of corresponding points in the second image (B x N x 2).
      mask: Optional mask to exclude points from choosing the best solution (B x N).

    Returns:
      A tuple containing:
        - R: Rotation matrix (B x 3 x 3).
        - t: Translation vector (B x 3).
        - points_3d: Triangulated 3D points (B x N x 3).
    """

    if E_mat.shape != (E_mat.shape[0], 3, 3):
        raise ValueError("Essential matrix must be a 3x3 matrix.")

    if K1.shape != (K1.shape[0], 3, 3) or K2.shape != (K2.shape[0], 3, 3):
        raise ValueError("Camera matrices must be 3x3 matrices.")

    if x1.shape != (x1.shape[0], x1.shape[1], 2) or x2.shape != (x2.shape[0], x2.shape[1], 2):
        raise ValueError("Point coordinates must be 2D matrices.")

    if mask is not None and mask.shape != (mask.shape[0], mask.shape[1]):
        raise ValueError("Mask must be a 2D matrix.")

    # Handle batch dimensions
    B = E_mat.shape[0]

    # Convert inputs to numpy arrays
    E_mat = E_mat.detach().cpu().numpy()
    K1 = K1.detach().cpu().numpy()
    K2 = K2.detach().cpu().numpy()
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()

    # Triangulate points
    _, R, t, _ = cv2.recoverPose(E_mat, K1, x1, K2, x2)

    # Reshape outputs to match batch dimensions
    R = np.reshape(R, (B, 3, 3))
    t = np.reshape(t, (B, 3))

    # Choose the best solution based on mask (if provided)
    if mask is not None:
        mask = mask.detach().cpu().numpy().astype(bool)
        R_best = None
        t_best = None
        points_3d_best = None

        for i in range(B):
            R_i = R[i]
            t_i = t[i]

            # Triangulate points using the current solution
            points_3d_i = cv2.triangulatePoints(K1[i], R_i @ K2[i], x1[i], x2[i]).T[:, :3]

            # Mask points and select the best solution
            if np.all(mask[i]):
                R_best = R_i
                t_best = t_i
                points_3d_best = points_3d_i
            else:
                mask_i = mask[i]
                R_i = R_i[mask_i, mask_i]
                t_i = t_i[mask_i]
                points_3d_i = points_3d_i[mask_i, :]

                _, R_best_i, t_best_i, _ = cv2.recoverPose(E_mat[i][mask_i], K1[i][mask_i], x1[i][mask_i], K2[i][mask_i], x2[i][mask_i])

                if R_best_i is not None:
                    R_best = R_best if R_best_i.trace() > R_i.trace() else R_i
                    t_best = t_best if R_best_i.trace() > R_i.trace() else t_i
                    points_3d_best = points_3d_best if R_best_i.trace() > R_i.trace() else points_3d_i

        R = np.reshape(R_best, (B, 3, 3))
        t = np.reshape(t_best, (B, 3))
        points_3d = np.reshape(points_3d_best, (B, -1, 3))

    # Convert outputs to tensors
    R = torch.from_numpy(R).float().to(E_mat.device)
    t = torch.from_numpy(t).float().to(E_mat.device)
    points_3d = torch.from_numpy(points_3d).float().to(E_mat.device)

    return R, t, points_3d

if __name__ == "__main__":
    # Create sample input values
    B = 2
    N = 5
    E_mat = np.random.randn(B, 3, 3)
    K1 = np.eye(3)
    K2 = np.eye(3)
    x1 = np.random.randn(B, N, 2)
    x2 = np.random.randn(B, N, 2)

    # Call the function
    R, t, points_3d = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)

    # Print the results
    print("Rotation matrix:")
    print(R)

    print("Translation vector:")
    print(t)

    print("Triangulated 3D points:")
    print(points_3d)