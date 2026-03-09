import numpy as np

def get_perspective_transform(points_src, points_dst):
    B = points_src.shape[0]
    A = []

    for i in range(B):
        src = points_src[i]
        dst = points_dst[i]
        A.append([
            [-src[0], -src[1], -1, 0, 0, 0, src[0] * dst[0], src[1] * dst[0], dst[0]],
            [0, 0, 0, -src[0], -src[1], -1, src[0] * dst[1], src[1] * dst[1], dst[1]]
        ])
        A.append([
            [-src[0], -src[1], -1, 0, 0, 0, src[0] * dst[2], src[1] * dst[2], dst[2]],
            [0, 0, 0, -src[0], -src[1], -1, src[0] * dst[3], src[1] * dst[3], dst[3]]
        ])
    
    A = np.array(A).reshape(-1, 9)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    return H

if __name__ == "__main__":
    points_src = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    points_dst = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]])
    
    H = get_perspective_transform(points_src, points_dst)
    print("Perspective Transform Matrix:\n", H)