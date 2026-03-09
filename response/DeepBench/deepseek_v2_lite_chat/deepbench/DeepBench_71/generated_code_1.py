import numpy as np
import tensorflow as tf

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    """
    Approximates Cauchy kernel in 1d.
    :param sigma: int, defining standard deviation of kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    dim = sigma
    x = tf.range(-dim, dim, dtype=tf.float32)
    kernel = (sigma / (np.pi * (1 + ((x / sigma)**2))))
    return kernel

if __name__ == "__main__":
    sigma = 1
    kernel = cauchy_kernel1d(sigma)
    print(kernel.numpy())