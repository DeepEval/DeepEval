import numpy as np

def get_perspective_transform(points_src, points_dst):
  """Calculates a perspective transform from four pairs of corresponding points.

  Args:
    points_src: Coordinates of quadrangle vertices in the source image with shape (B, 4, 2).
    points_dst: Coordinates of the corresponding quadrangle vertices in the destination image with shape (B, 4, 2).

  Returns:
    The perspective transformation with shape (B, 3, 3).
  """

  # Extract points
  x_src, y_src = points_src[:, 0], points_src[:, 1]
  x_dst, y_dst = points_dst[:, 0], points_dst[:, 1]

  # Calculate constants
  const = np.ones_like(x_src)
  A = np.stack([x_src, y_src, const, np.zeros_like(x_src)], axis=-1)
  B = np.stack([np.zeros_like(x_src), np.zeros_like(x_src), np.zeros_like(x_src), const], axis=-1)
  C = np.stack([-x_dst * x_src, -x_dst * y_src, -x_dst * const, x_src], axis=-1)
  D = np.stack([-y_dst * x_src, -y_dst * y_src, -y_dst * const, y_src], axis=-1)

  # Calculate the transformation matrix
  M = np.linalg.solve(np.concatenate([A, B, C, D], axis=0),
                        np.concatenate([-y_dst * x_src, -y_dst * y_src, -y_dst * const, x_src, y_src, const], axis=0))
  H = M[:3, :] / M[3, :]

  return H

if __name__ == "__main__":
  # Create sample input values
  points_src = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32).reshape(1, 4, 2)
  points_dst = np.array([[120, 80], [250, 80], [250, 180], [120, 180]], dtype=np.float32).reshape(1, 4, 2)

  # Call the function and print the results
  H = get_perspective_transform(points_src, points_dst)
  print(H)