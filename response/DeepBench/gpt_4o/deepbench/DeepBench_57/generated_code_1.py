import tensorflow as tf
import numpy as np
from typing import Callable, Optional, Tuple

def mvnquad(
    func: Callable[[tf.Tensor], tf.Tensor],
    means: tf.Tensor,
    covs: tf.Tensor,
    H: int,
    Din: Optional[int] = None,
    Dout: Optional[Tuple[int, ...]] = None,
) -> tf.Tensor:
    if Din is None:
        Din = means.shape[1]
    if Din is None:
        raise ValueError(
            "If `Din` is passed as `None`, `means` must have a known shape. "
            "Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
            "is problematic. Consider using your own session."
        )
    
    gh_x, gh_w = np.polynomial.hermite.hermgauss(H)
    gh_x = gh_x.astype(np.float32)
    gh_w = gh_w.astype(np.float32) * np.pi**-0.5

    chol_covs = tf.linalg.cholesky(covs)
    gh_x_reshaped = tf.reshape(gh_x, [1, 1, -1, 1])
    gh_w_reshaped = tf.reshape(gh_w, [1, 1, -1])

    X = (
        means[:, None, :] +
        tf.matmul(chol_covs, gh_x_reshaped, transpose_b=True)
    )
    fX = func(tf.reshape(X, [-1, Din]))

    if Dout is None:
        Dout = fX.shape[1:]
    fX = tf.reshape(fX, [-1, H] + list(Dout))
    quadrature = tf.reduce_sum(fX * gh_w_reshaped[..., None], axis=1)
    
    return quadrature

if __name__ == "__main__":
    # Sample function that squares its input
    def sample_func(x):
        return x ** 2

    # Sample input values
    means = tf.constant([[1.0, 2.0], [2.0, 1.0]], dtype=tf.float32)
    covs = tf.constant([
        [[0.1, 0.0], [0.0, 0.1]],
        [[0.2, 0.0], [0.0, 0.2]]
    ], dtype=tf.float32)

    # Number of Gauss-Hermite points
    H = 10

    # Compute the quadrature
    result = mvnquad(sample_func, means, covs, H)
    print("Quadrature results:", result.numpy())