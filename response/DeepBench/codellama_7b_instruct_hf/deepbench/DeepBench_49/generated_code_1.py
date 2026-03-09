import numpy as np

def matrix_cofactor_tensor(matrix):
    # Check if all input matrices are singular (i.e., their determinants are zero)
    if np.linalg.det(matrix) == 0:
        raise Exception("All input matrices are singular")

    # Compute the cofactor matrix for each input matrix
    cofactor_matrix = np.empty((*matrix.shape[:-2], 3, 3), dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cofactor_matrix[i, j] = np.linalg.inv(matrix[i, j])

    return cofactor_matrix

if __name__ == "__main__":
    # Create sample input values
    matrix = np.random.rand(3, 3)

    # Call the function and print the results
    cofactor_matrix = matrix_cofactor_tensor(matrix)
    print(cofactor_matrix)