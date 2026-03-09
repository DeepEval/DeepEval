import numpy as np

def matrix_cofactor_tensor(matrix):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Input matrices must have shape (*, 3, 3)")

    dets = np.linalg.det(matrix)
    if np.all(dets == 0):
        raise Exception("All input matrices are singular")

    cofactor_matrices = np.empty_like(matrix)

    for idx in np.ndindex(matrix.shape[:-2]):
        m = matrix[idx]
        cofactor = np.empty((3, 3))
        
        cofactor[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
        cofactor[0, 1] = -(m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        cofactor[0, 2] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
        
        cofactor[1, 0] = -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1])
        cofactor[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
        cofactor[1, 2] = -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0])
        
        cofactor[2, 0] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
        cofactor[2, 1] = -(m[0, 0] * m[1, 2] - m[0, 2] * m[1, 0])
        cofactor[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        
        cofactor_matrices[idx] = cofactor

    return cofactor_matrices

if __name__ == "__main__":
    sample_matrices = np.array([
        [[1, 2, 3], [0, 1, 4], [5, 6, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[2, -1, 0], [1, 3, -1], [1, 0, 1]]
    ])

    try:
        cofactors = matrix_cofactor_tensor(sample_matrices)
        print("Cofactor Matrices:")
        print(cofactors)
    except Exception as e:
        print("An error occurred:", e)