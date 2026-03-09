import tensorflow as tf
from tensorflow import Tensor
from typing import Any

def check_shapes(*args: str):
    def decorator(func: Any):
        return func
    return decorator

@check_shapes("x: [D, broadcast N]", "mu: [D, broadcast N]", "L: [D, D]", "return: [N]")
def multivariate_normal(x: Tensor, mu: Tensor, L: Tensor) -> tf.Tensor:
    D = tf.shape(x)[0]  # Dimensionality of the data
    N = tf.shape(x)[1]  # Number of samples

    # Center the data
    x_centered = x - mu

    # Solve Lz = x_centered
    z = tf.linalg.cholesky_solve(L, x_centered)

    # Compute the log determinant of L
    log_det_L = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    # Compute the log-density
    log_density = -0.5 * (D * tf.math.log(2 * np.pi) + log_det_L + tf.reduce_sum(tf.square(z), axis=0))

    return log_density

if __name__ == "__main__":
    D = 3  # Dimensionality
    N = 5  # Number of samples

    x = tf.random.normal((D, N))
    mu = tf.random.normal((D, N))
    L = tf.linalg.cholesky(tf.eye(D))  # Identity matrix as a simple covariance

    log_densities = multivariate_normal(x, mu, L)
    print(log_densities.numpy())