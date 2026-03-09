import numpy as np
import tensorflow as tf

@tf.function
def mvnquad(func, means, covs, H, Din=None, Dout=None):
    # Figure out input shape information
    if Din is None:
        Din = means.shape[1]
    if Din is None:
        raise ValueError("If `Din` is passed as `None`, `means` must have a known shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` is problematic. Consider using your own session.")
    
    # Compute the Cholesky decomposition of the covariance matrices
    covs_cholesky = tf.linalg.cholesky(covs)
    
    # Compute the Gauss-Hermite evaluation points and weights
    eval_points, weights = tf.compat.v1.tfd.MultivariateNormalDiag(
        loc=tf.zeros([Din]),
        scale_diag=tf.ones([Din])
    ).sample_hermite([H])
    
    # Compute the function values at the Gauss-Hermite evaluation points
    func_values = func(means + tf.matmul(covs_cholesky, eval_points))
    
    # Compute the expectation integrals
    expectation_values = tf.reduce_sum(func_values * weights, axis=0)
    
    return expectation_values

if __name__ == "__main__":
    # Create sample input values
    means = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    covs = tf.constant([[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]]])
    func = lambda x: tf.reduce_sum(x**2, axis=1)
    H = 10
    Din = None
    Dout = None
    
    # Call the function and print the results
    result = mvnquad(func, means, covs, H, Din, Dout)
    print(result)