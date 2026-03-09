import numpy as np
import cv2

def depth_to_3d(depth, camera_matrix, normalize=False):
    """
    Computes a 3D point per pixel based on the depth value and camera intrinsics.

    Args:
        depth (np.array): A tensor representing depth values per pixel, with shape (B, 1, H, W).
        camera_matrix (np.array): A tensor containing camera intrinsics, with shape (B, 3, 3).
        normalize (bool): Whether to normalize the 3D points to the range [-1, 1].

    Returns:
        np.array: A tensor with a 3D point per pixel of the same resolution as the input, with shape (B, 3, H, W).
    """
    # Check input types and shapes
    if not isinstance(depth, np.ndarray):
        raise TypeError("Depth must be a numpy array.")
    if not isinstance(camera_matrix, np.ndarray):
        raise TypeError("Camera matrix must be a numpy array.")
    if depth.ndim != 4:
        raise ValueError("Depth must have 4 dimensions.")
    if camera_matrix.ndim != 3:
        raise ValueError("Camera matrix must have 3 dimensions.")
    if depth.shape[0] != camera_matrix.shape[0]:
        raise ValueError("Depth and camera matrix must have the same batch size.")
    if depth.shape[1] != 1:
        raise ValueError("Depth must have a single channel.")
    if depth.shape[2] != camera_matrix.shape[1]:
        raise ValueError("Depth and camera matrix must have the same height.")
    if depth.shape[3] != camera_matrix.shape[2]:
        raise ValueError("Depth and camera matrix must have the same width.")

    # Compute 3D points
    B, _, H, W = depth.shape
    points = np.empty((B, 3, H, W), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                z = depth[b, 0, h, w]
                if z == 0:
                    points[b, :, h, w] = np.array([0, 0, 0])
                else:
                    p = camera_matrix[b, :, :] @ np.array([w, h, 1])
                    points[b, :, h, w] = p / z

    # Normalize points if requested
    if normalize:
        points = points / (2 * np.max(points))

    return points

if __name__ == "__main__":
    # Create sample input values
    B = 1
    H = 2
    W = 3
    depth = np.random.rand(B, 1, H, W)
    camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Call function and print results
    points = depth_to_3d(depth, camera_matrix)
    print(points)