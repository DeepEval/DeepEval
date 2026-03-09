import numpy as np

def run_7point(points1, points2):
    assert points1.shape == (points1.shape[0], 7, 2), "points1 must have shape (B, 7, 2)"
    assert points2.shape == (points2.shape[0], 7, 2), "points2 must have shape (B, 7, 2)"
    
    B = points1.shape[0]
    F_matrices = []

    for b in range(B):
        pts1 = points1[b]
        pts2 = points2[b]

        # Normalize points
        def normalize(points):
            centroid = np.mean(points, axis=0)
            scale = np.sqrt(2) / np.mean(np.linalg.norm(points - centroid, axis=1))
            T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])
            normalized_points = np.dot(T, np.hstack((points, np.ones((points.shape[0], 1)))).T).T
            return normalized_points[:, :2], T
        
        norm_pts1, T1 = normalize(pts1)
        norm_pts2, T2 = normalize(pts2)

        # Construct the linear system
        A = np.zeros((7, 9))
        for i in range(7):
            x1, y1 = norm_pts1[i]
            x2, y2 = norm_pts2[i]
            A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        F1 = Vt[-1].reshape(3, 3)
        F2 = Vt[-2].reshape(3, 3)

        # Form cubic polynomial for determinant
        coeffs = np.array([F1[0, 0], F1[0, 1], F1[0, 2], F1[1, 0], F1[1, 1], F1[1, 2], F1[2, 0], F1[2, 1], F1[2, 2]])
        coeffs = np.append(coeffs, np.array([F2[0, 0], F2[0, 1], F2[0, 2], F2[1, 0], F2[1, 1], F2[1, 2], F2[2, 0], F2[2, 1], F2[2, 2]]))

        # Cubic polynomial coefficients
        cubic_coeffs = np.array([0, 0, 0, 0])
        for i in range(0, len(coeffs), 3):
            if i < len(coeffs) - 1:
                cubic_coeffs = np.append(cubic_coeffs, coeffs[i])

        # Find roots of the cubic polynomial
        roots = np.roots(cubic_coeffs)
        
        F_matrices_batch = []
        for root in roots:
            if np.iscomplex(root):
                continue
            F = F1 * root + F2 * (1 - root)
            F_matrices_batch.append(F)
        
        F_matrices.append(np.array(F_matrices_batch))

    return np.concatenate(F_matrices, axis=0).reshape(B, -1, 3)

if __name__ == "__main__":
    points1 = np.random.rand(1, 7, 2)
    points2 = np.random.rand(1, 7, 2)
    F_matrices = run_7point(points1, points2)
    print("Computed Fundamental Matrices:\n", F_matrices)