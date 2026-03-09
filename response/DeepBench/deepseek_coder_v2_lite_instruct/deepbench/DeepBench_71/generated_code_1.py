import tensorflow as tf
import numpy as np

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    assert sigma > 0
    dim = 2 * sigma + 1
    x = np.linspace(-sigma, sigma, dim)
    kernel = 1 / (np.pi * (sigma**2 + x**2))
    return tf.convert_to_tensor(kernel, dtype=tf.float32)

if __name__ == "__main__":
    sigma = 2
    kernel = cauchy_kernel1d(sigma)
    print(kernel.numpy())