import numpy as np
from cv2 import solve

def essential_to_rotation_and_translation(E):
    """
    Convert an essential matrix to a rotation and translation matrix.
    """
    assert E.shape == (3, 3)
    assert np.linalg.det(E) >= 0
    Rt = cv2.decomposeEssentialMat(E)
    R, t = Rt[0][:3, :3], Rt[1][:3]
    return R, t

def essential_matrix_from_fundamental_matrices(F1, F2):
    """
    Given two fundamental matrices, recover the essential matrix.
    """
    assert F1.shape == (3, 3)
    assert F2.shape == (3, 3)
    E = F1.T.dot(F2)
    return E

def fundamental_to_essential_matrix(F1, F2):
    """
    Convert two fundamental matrices to an essential matrix.
    """
    assert F1.shape == (3, 3)
    assert F2.shape == (3, 3)
    E = F1.dot(F2.T)
    return E

def project_points(P, X):
    """
    Project a set of 3D points into the image plane using a camera matrix.
    """
    assert P.shape == (4, 4)
    assert X.shape == (N, 3) or X.shape == (N, 4)
    P = P[:3, :4]
    X = X if X.shape[-1] == 3 else np.hstack([X, np.ones((X.shape[0], 1))])
    PX = P.dot(X)
    z = PX[2, :]
    u = PX[0, :]/z if z > 0 else np.inf
    v = PX[1, :]/z if z > 0 else np.inf
    u, v = int(np.round(u)), int(np.round(v))
    return u, v, int(np.round(z))

def triangulate_points(x1, x2, R1, R2, t1, t2, mask=None):
    """
    Triangulate points using two views and the camera matrices.
    """
    assert x1.shape == (N1, 3) and x2.shape == (N2, 3)
    assert R1.shape == (3, 3) and R2.shape == (3, 3)
    assert t1.shape == (3,) and t2.shape == (3,)
    assert mask is None or mask.shape == (N1 + N2, 1)
    
    X = np.hstack([x1, x2])
    if mask is not None:
        X = X[mask]
        R1 = R1[mask]
        R2 = R2[mask]
        t1 = t1[mask]
        t2 = t2[mask]
    
    A = R2.T.dot(R1)
    A = np.hstack([A, -R2.T.dot(t1)])
    A = np.hstack([A, R1.T.dot(t2)])
    A = np.hstack([A, -t1.T.dot(t2)])
    A = np.linalg.inv(K2).dot(A)
    A = np.linalg.solve(A)
    P = K1.dot(R1).dot(A[:3])
    P = np.vstack([P, [0, 0, 0, 1]])
    x = P[:3, -1] / P[:3, -1][3]
    x = x / x[2]
    x = K1.dot(R1).dot(x)
    x = x / x[2]
    return x

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    """
    Recover the relative camera rotation and translation from an estimated essential matrix.
    """
    E = essential_matrix_from_fundamental_matrices(K1.dot(x1), K2.dot(x2))
    assert np.allclose(E_mat, E)
    
    E_inv = np.linalg.inv(E)
    F1 = E_inv.dot(K2.dot(x1.T)).T
    F2 = E_inv.dot(K1.dot(x2.T)).T
    F1, F2 = F1 + np.eye(3), F2 + np.eye(3)
    
    R1, t1 = essential_to_rotation_and_translation(E_inv)
    R2, t2 = essential_to_rotation_and_translation(E)
    
    x = triangulate_points(x1, x2, R1, R2, t1, t2, mask)
    return R1, t1, x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Sample input values
    N1, N2 = 50, 50
    x1 = np.random.rand(N1, 3)
    x2 = np.random.rand(N2, 3)
    K1 = np.eye(3)
    K2 = np.eye(3)
    E_mat = np.eye(3)
    
    # Call the function
    R1, t1, x = motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2)
    
    # Plot results
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c='r', marker='o')
    ax1.set_title('Triangulated points')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c='b', marker='o')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('3D points');
    
    plt.show()