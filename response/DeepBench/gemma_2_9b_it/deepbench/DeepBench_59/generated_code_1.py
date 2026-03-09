import numpy as np

def get_perspective_transform(points_src, points_dst):
    B, _, _ = points_src.shape
    src_homo = np.hstack((points_src, np.ones((B, 4, 1))))
    dst_homo = np.hstack((points_dst, np.ones((B, 4, 1))))
    A = np.zeros((B * 4, 9))
    for i in range(B):
        for j in range(4):
            A[i * 4 + j, :] = src_homo[i, j, :]

if __name__ == "__main__":
    print("a")