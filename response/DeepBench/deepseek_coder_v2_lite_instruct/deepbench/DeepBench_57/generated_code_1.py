import tensorflow as tf
from typing import Callable, Optional, Tuple

def mvnquad(func: Callable[[tf.Tensor], tf.Tensor], means: tf.Tensor, covs: tf.Tensor, H: int, Din: Optional[int] = None, Dout: Optional[Tuple[int, ...]] = None) -> tf.Tensor:
    if Din is None:
        Din = means.shape[1]
    if Din is None:
        raise ValueError("If `Din` is passed as `None`, `means` must have a known shape. " "Running mvnquad in `autoflow` without specifying `Din` and `Dout` " "is problematic. Consider using your own session.")

    # Check shapes
    if means.shape[0] != covs.shape[0]:
        raise ValueError("The first dimension of `means` and `covs` must be the same.")
    if covs.shape[1] != Din or covs.shape[2] != Din:
        raise ValueError("The shape of `covs` is incorrect. It should be [N, Din, Din].")

    N = means.shape[0]
    if Dout is None:
        Dout = ()

    # Initialize output tensor
    results = tf.TensorArray(tf.float32, size=N)

    for i in tf.range(N):
        mean = means[i]
        cov = covs[i]
        weights, points = tf.linalg.eigh(cov)
        scaled_points = tf.matmul(points, tf.reshape(tf.sqrt(weights), [-1, 1]))

        def gauss_hermite_quad(x):
            return func(x) * tf.exp(-tf.reduce_sum(x**2, axis=-1, keepdims=True))

        integral = tf.zeros([H] + list(Dout))
        for j in tf.range(H):
            point = tf.expand_dims(scaled_points[j], 0)
            integral += gauss_hermite_quad(mean + point)

        integral /= tf.sqrt(tf.reduce_prod(tf.linalg.diag_part(cov))) * (tf.math.sqrt(tf.constant(2.0))**Din)
        results = results.write(i, integral)

    return results.stack()

if __name__ == "__main__":
    # Define a sample function
    def sample_func(x):
        return tf.reduce_sum(x**2, axis=-1)

    # Create sample input values
    N = 3
    Din = 2
    H = 5
    means = tf.random.normal([N, Din])
    covs = tf.linalg.diag(tf.ones([N, Din]))

    # Call the function
    results = mvnquad(sample_func, means, covs, H)

    # Print the results
    print(results.numpy())