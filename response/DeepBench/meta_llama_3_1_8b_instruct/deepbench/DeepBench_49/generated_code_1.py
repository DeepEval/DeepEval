import numpy as np
import tensorflow as tf

def matrix_cofactor_tensor(matrix):
    # Calculate the determinant of each matrix in the tensor
    determinants = tf.map_fn(lambda x: tf.linalg.det(x), matrix)

    # Check if any of the determinants are zero
    singular_matrices = tf.math.count_nonzero(determinants == 0)

    if singular_matrices == tf.size(determinants):
        raise Exception("All input matrices are singular.")

    # Calculate the cofactor matrix for each matrix in the tensor
    cofactors = tf.map_fn(lambda x: tf.stack([tf.linalg.det(tf.slice(x, [0, i], [-1, -1])), 
                                             tf.linalg.det(tf.slice(x, [0, -1], [-1, i])), 
                                             tf.linalg.det(x - tf.slice(x, [0, i], [-1, -1]) * tf.slice(x, [i, 0], [-1, -1]))], axis=1), 
                          matrix)

    return cofactors

if __name__ == "__main__":
    # Create sample input values
    matrix = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 10]], 
                       [[2, 3, 4], [5, 6, 7], [8, 9, 11]], 
                       [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
    matrix = tf.constant(matrix, dtype=tf.float64)

    # Call the function and print the results
    try:
        cofactors = matrix_cofactor_tensor(matrix)
        print(cofactors)
    except Exception as e:
        print(e)