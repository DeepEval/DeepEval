import numpy as np

def matrix_cofactor_tensor(matrix):
    det = np.linalg.det(matrix)
    if np.all(det == 0):
        raise Exception("All input matrices are singular.")
    
    cofactor = np.empty(matrix.shape)
    for i in range(matrix.shape[0]):
        m = matrix[i]
        cofactor[i] = np.array([
            [m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1], m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2], m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]],
            [m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2], m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0], m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]],
            [m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0], m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1], m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]]
        ])
    
    return cofactor

if __name__ == "__main__":
    matrices = np.array([[[2, 3, 1], [1, 2, 1], [1, 0, 1]],
                         [[1, 1, 1], [0, 0, 0], [1, 2, 3]],
                         [[4, 5, 6], [7, 8, 9], [1, 2, 3]]])
    
    try:
        cofactors = matrix_cofactor_tensor(matrices)
        print("Cofactor matrices:\n", cofactors)
    except Exception as e:
        print(e)