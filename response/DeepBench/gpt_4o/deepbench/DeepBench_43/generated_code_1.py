import tensorflow as tf
import tensorflow_probability as tfp
from tensor_annotations import check_shapes
from tensor_annotations.tensorflow import TensorType
from typing import Any

@check_shapes(
    "x: [D, broadcast N]",
    "mu: [D, broadcast N]",
    "L: [D, D]",
    "return: [N]",
)
def multivariate_normal(x: TensorType, mu: TensorType, L: TensorType) -> tf.Tensor:
    # Compute the dimension D
    D = tf.shape(L)[0]

    # Solve for L^{-1} (x - mu) using Cholesky factor L
    L_inv_x_minus_mu = tf.linalg.triangular_solve(L, x - mu)

    # Compute the Mahalanobis term
    mahalanobis_term = tf.reduce_sum(tf.square(L_inv_x_minus_mu), axis=0)

    # Compute the log determinant of the covariance matrix using the Cholesky factor
    log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    # Compute the log density
    log_density = -0.5 * (D * tf.math.log(2.0 * tf.constant(tf.pi)) + log_det_cov + mahalanobis_term)

    return log_density

if __name__ == "__main__":
    # Define the dimension D and number of samples N
    D = 3
    N = 2

    # Create sample input values
    x = tf.constant([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], dtype=tf.float32)  # Shape [D, N]
    mu = tf.constant([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=tf.float32)  # Shape [D, N]
    L = tf.linalg.cholesky(tf.constant([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=tf.float32))  # Shape [D, D]

    # Call the multivariate_normal function
    log_densities = multivariate_normal(x, mu, L)

    # Print the results
    print("Log densities:", log_densities.numpy())