import numpy as np

def get_perspective_transform(points_src, points_dst):
    B = points_src.shape[0]
    A = np.zeros((B, 8, 8))
    b = np.zeros((B, 8, 1))

    for i in range(B):
        for j in range(4):
            x_src, y_src = points_src[i, j]
            x_dst, y_dst = points_dst[i, j]
            A[i, 2 * j] = [x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src]
            A[i, 2 * j + 1] = [0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src]
            b[i, 2 * j] = x_dst
            b[i, 2 * j + 1] = y_dst

    h = np.linalg.solve(A, b)
    h = np.concatenate([h, np.ones((B, 1, 1))], axis=1).reshape(B, 3, 3)

    return h

if __name__ == "__main__":
    points_src = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
    points_dst = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], dtype=np.float32)

    transform_matrix = get_perspective_transform(points_src, points_dst)
    print("Perspective Transformation Matrix:")
    print(transform_matrix)