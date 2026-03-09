import tensorflow as tf
import numpy as np

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    assert sigma > 0
    dim = 100  # Define the dimension of the kernel
    x = tf.range(-dim // 2, dim // 2, dtype=tf.float32)
    kernel = 1.0 / (1.0 + (x / sigma) ** 2)
    return kernel / tf.reduce_sum(kernel)  # Normalize the kernel

if __name__ == "__main__":
    sigma = 1
    kernel = cauchy_kernel1d(sigma)
    print("Cauchy Kernel (1D) with sigma =", sigma)
    print(kernel.numpy())