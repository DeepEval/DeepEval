import tensorflow as tf

def matrix_cofactor_tensor(matrix):
    num_matrices = matrix.shape[-2].value  # Assuming the last dimension is for matrices
    cofactor_matrices = []

    for i in range(num_matrices):
        # Compute determinant
        determinant = tf.linalg.det(matrix[..., i])
        # Check if determinant is zero
        if tf.math.is_nan(determinant):
            raise Exception("Matrix is singular.")
        # Compute cofactor matrix
        cofactor = tf.linalg.inv(matrix[..., i]) * determinant * tf.constant([[(i % 2 == 0) - (i % 2 == 1)], [-(i // 2) * (i % 2 == 0)], [(i // 2) * (i % 2 == 1)]])
        cofactor_matrices.append(cofactor)

    return tf.stack(cofactor_matrices, axis=-2)


if __name__ == "__main__":
    # Sample input values
    matrix_1 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    matrix_2 = tf.constant([[[9, 8, 7], [6, 5, 4], [3, 2, 1]]])

    # Call the function
    cofactor_matrix = matrix_cofactor_tensor(tf.stack([matrix_1, matrix_2]))

    # Print the results
    print("Cofactor matrix for matrix 1:")
    print(cofactor_matrix[..., 0])
    print("Cofactor matrix for matrix 2:")
    print(cofactor_matrix[..., 1])