import numpy as np
from scipy.linalg import svd
from scipy.optimize import roots

def run_7point(points1, points2):
    # Check input shapes
    assert points1.shape == points2.shape, "Input tensors must have the same shape"
    assert points1.shape[1] == 7, "Number of points must be exactly 7"
    assert points1.shape[2] == 2, "Points must be 2D"
    B, N, _ = points1.shape

    # Normalize points
    points1 = points1 / np.linalg.norm(points1, axis=2, keepdims=True)
    points2 = points2 / np.linalg.norm(points2, axis=2, keepdims=True)

    # Construct linear system
    A = np.zeros((14, 9))
    for i in range(7):
        A[2*i, 0] = points1[i, 0]
        A[2*i, 1] = points1[i, 1]
        A[2*i, 2] = 1
        A[2*i, 3] = -points1[i, 0]*points2[i, 0]
        A[2*i, 4] = -points1[i, 1]*points2[i, 0]
        A[2*i, 5] = -points2[i, 0]
        A[2*i, 6] = -points1[i, 0]*points2[i, 1]
        A[2*i, 7] = -points1[i, 1]*points2[i, 1]
        A[2*i, 8] = -points2[i, 1]
        A[2*i+1, 0] = points1[i, 1]
        A[2*i+1, 1] = -points1[i, 0]
        A[2*i+1, 2] = 0
        A[2*i+1, 3] = -points1[i, 1]*points2[i, 0]
        A[2*i+1, 4] = points1[i, 0]*points2[i, 0]
        A[2*i+1, 5] = 0
        A[2*i+1, 6] = -points1[i, 1]*points2[i, 1]
        A[2*i+1, 7] = points1[i, 0]*points2[i, 1]
        A[2*i+1, 8] = 0

    # Solve linear system using SVD
    U, s, Vt = svd(A, full_matrices=False)
    S = np.diag(s)
    F1 = np.dot(U, np.dot(S, Vt))
    F2 = np.dot(U, np.dot(S[:, ::-1], Vt))

    # Form cubic polynomial
    x = np.linspace(-10, 10, 100)
    a = np.zeros((B, 3))
    a[:, 0] = F1[:, 8]
    a[:, 1] = F1[:, 7] - F2[:, 7]
    a[:, 2] = F1[:, 6] - F2[:, 6]
    b = np.zeros((B, 3))
    b[:, 0] = 0
    b[:, 1] = 2*F1[:, 6]
    b[:, 2] = 2*F2[:, 6]
    c = np.zeros((B, 3))
    c[:, 0] = 0
    c[:, 1] = 0
    c[:, 2] = F1[:, 0]*F2[:, 0] - F1[:, 1]*F2[:, 1]

    # Solve for roots of cubic polynomial
    roots_out = []
    for i in range(B):
        r = roots([a[i, 0], b[i, 0], c[i, 0], d[i, 0]])
        roots_out.append(r[np.isreal(r)][0].real)

    # Compute potential fundamental matrices
    m = 0
    F_out = np.zeros((B, 3*len(roots_out), 3))
    for i in range(B):
        r = roots_out[i]
        for j in range(len(r)):
            if np.isreal(r[j]):
                m += 1
                F_out[i, m-1, 0] = F1[i, 0] + r[j]*F1[i, 6]
                F_out[i, m-1, 1] = F1[i, 1] + r[j]*F1[i, 7]
                F_out[i, m-1, 2] = F1[i, 2] + r[j]*F1[i, 8]

    return F_out

if __name__ == "__main__":
    np.random.seed(0)
    points1 = np.random.rand(2, 7, 2)
    points2 = np.random.rand(2, 7, 2)
    F = run_7point(points1, points2)
    print(F)