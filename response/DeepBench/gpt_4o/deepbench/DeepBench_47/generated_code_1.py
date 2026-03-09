import numpy as np

def normalize_points(points):
    mean = np.mean(points, axis=1, keepdims=True)
    std = np.std(points, axis=1, keepdims=True)
    transform = np.array([[[1/std[i, 0, 0], 0, -mean[i, 0, 0]/std[i, 0, 0]], 
                           [0, 1/std[i, 0, 1], -mean[i, 0, 1]/std[i, 0, 1]], 
                           [0, 0, 1]] for i in range(points.shape[0])])
    points_norm = np.einsum('bij,bnj->bni', transform, np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=-1))
    return points_norm, transform

def construct_A(points1, points2):
    A = np.empty((points1.shape[0], 7, 9))
    for i in range(7):
        A[:, i] = np.einsum('bi,bj->bij', points2[:, i], points1[:, i]).reshape(-1, 9)
    return A

def solve_7point(A):
    U, S, Vt = np.linalg.svd(A)
    F1 = Vt[:, -1, :].reshape(-1, 3, 3)
    F2 = Vt[:, -2, :].reshape(-1, 3, 3)
    
    return F1, F2

def run_7point(points1, points2):
    assert points1.shape == points2.shape
    assert points1.shape[1] == 7 and points1.shape[2] == 2
    
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)
    
    A = construct_A(points1_norm, points2_norm)
    F1, F2 = solve_7point(A)
    
    F_candidates = []
    for i in range(points1.shape[0]):
        fun = lambda alpha: np.linalg.det(alpha * F1[i] + (1 - alpha) * F2[i])
        
        coeffs = [fun(1), -3*fun(2/3), 3*fun(1/3), -fun(0)]
        
        roots = np.roots(coeffs)
        roots = np.real(roots[np.isreal(roots)])
        
        F_matrices = []
        for root in roots:
            F_approx = root * F1[i] + (1 - root) * F2[i]
            F_approx = np.dot(T2[i].T, np.dot(F_approx, T1[i]))
            F_matrices.append(F_approx / F_approx[2, 2])
        
        F_candidates.append(np.array(F_matrices))
    
    max_F_count = max(len(f) for f in F_candidates)
    F_result = np.zeros((points1.shape[0], max_F_count * 3, 3))
    
    for i, F_set in enumerate(F_candidates):
        F_result[i, :len(F_set) * 3, :] = np.vstack(F_set)
    
    return F_result

if __name__ == "__main__":
    points1 = np.array([[[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9], [1.0, 1.0], [1.1, 1.1]]])
    points2 = np.array([[[0.6, 0.5], [0.7, 0.6], [0.8, 0.7], [0.9, 0.8], [1.0, 0.9], [1.1, 1.0], [1.2, 1.1]]])
    fundamental_matrices = run_7point(points1, points2)
    print(fundamental_matrices)