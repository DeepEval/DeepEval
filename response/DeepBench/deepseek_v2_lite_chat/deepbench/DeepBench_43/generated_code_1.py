import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def cholesky_solve(L, x):
    L = tf.linalg.LinearOperatorLowerTriangular(L).matmul(x)
    return L

def multivariate_normal(x: tf.Tensor, mu: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    # Compute the number of samples
    N = math_ops.cast(K.shape(x)[1], tf.float32)
    
    # Check shapes
    assert math_ops.reduce_all(math_ops.less_equal(K.shape(mu)[1], K.shape(L)[1])),
                             "mu: [D, broadcast N] should have shape equal to L: [D, D]")
    
    # Check if L is a valid Cholesky decomposition
    assert math_ops.reduce_all(math_ops.greater_equal(tf.linalg.cholesky(L), 0)), "L should be a valid Cholesky decomposition"
    
    # Compute the log determinant of L
    log_det_L = tf.linalg.logdet(L)
    
    # Compute the difference between x and mu
    diff = x - mu
    
    # Compute the log-density using properties of multivariate normal
    log_density = -N / 2 * (math_ops.log(2 * math_ops.pi) + log_det_L + tf.reduce_sum(diff**2, axis=1))
    
    return log_density

if __name__ == "__main__":
    # Example usage
    with ops.device("/CPU:0"):  # Ensure the device is set to CPU for simplicity
        # Create sample input values
        mu = tf.constant([[0.], [0.]], dtype=tf.float32)
        L = tf.constant([[1., 0.5], [0.5, 2.]], dtype=tf.float32)
        x = tf.random.normal([2, 10], mean=mu, stddev=1.)

        # Call the function
        log_density = multivariate_normal(x, mu, L)

        # Print the results
        print("Log-density at x:", log_density)