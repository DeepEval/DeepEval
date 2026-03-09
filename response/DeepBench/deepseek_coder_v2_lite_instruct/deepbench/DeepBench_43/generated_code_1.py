import tensorflow as tf
from tensorflow.python.framework.ops import TensorType

def multivariate_normal(x: TensorType, mu: TensorType, L: TensorType) -> tf.Tensor:
    """ Computes the log-density of a multivariate normal. """
    d = tf.shape(mu)[0]
    x_centered = x - mu
    z = tf.linalg.triangular_solve(L, x_centered, lower=True)
    z_squared = tf.reduce_sum(tf.square(z), axis=0, keepdims=True)
    log_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
    log_density = -0.5 * d * tf.math.log(2 * tf.constant(3.141592653589793, dtype=tf.float32)) - 0.5 * z_squared - log_det
    return tf.squeeze(log_density, axis=0)

if __name__ == "__main__":
    # Example usage
    mu = tf.constant([1.0, 2.0], dtype=tf.float32)
    L = tf.linalg.cholesky(tf.constant([[4.0, 0.0], [0.0, 9.0]], dtype=tf.float32))
    x = tf.constant([[1.0, 2.1], [0.9, 2.0]], dtype=tf.float32)
    log_densities = multivariate_normal(x, mu, L)
    print(log_densities.numpy())