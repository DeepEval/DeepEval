import numpy as np
import tensorflow as tf
from typing import Callable, Optional, Tuple

TensorType = tf.Tensor

def mvnquad(func: Callable[[tf.Tensor], tf.Tensor], means: TensorType, covs: TensorType, H: int, Din: Optional[int] = None, Dout: Optional[Tuple[int, ...]] = None) -> tf.Tensor:
    if Din is None:
        Din = means.shape[1]
    if Dout is None:
        Dout = ()

    # Get the number of samples
    N = means.shape[0]

    # Prepare Gauss-Hermite quadrature points and weights
    x, w = np.polynomial.hermite.hermgauss(H)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    w = tf.convert_to_tensor(w, dtype=tf.float32)

    # Create a placeholder for results
    results = []

    # Loop over each mean and covariance
    for i in range(N):
        mean = means[i]
        cov = covs[i]

        # Cholesky decomposition to obtain samples from the multivariate normal
        L = tf.linalg.cholesky(cov)
        z = tf.reshape(x, (-1, 1)) @ tf.reshape(mean, (1, -1)) + tf.matmul(tf.reshape(x, (-1, 1)), L)
        z = tf.reshape(z, (-1, Din))

        # Evaluate the function
        output = func(z)

        # Weight the output by the quadrature weights
        weighted_output = tf.reduce_sum(output * w[:, None], axis=0)
        results.append(weighted_output)

    return tf.stack(results)

def sample_function(x):
    return tf.reduce_sum(x, axis=1, keepdims=True)

if __name__ == "__main__":
    N = 3
    Din = 2
    Dout = (1,)
    H = 5

    means = tf.constant(np.random.rand(N, Din), dtype=tf.float32)
    covs = tf.constant(np.random.rand(N, Din, Din), dtype=tf.float32)
    covs = covs @ tf.transpose(covs, perm=[0, 2, 1])  # Making them positive semi-definite

    result = mvnquad(sample_function, means, covs, H, Din, Dout)
    print(result.numpy())