import tensorflow as tf
import numpy as np

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    # Create a 1D tensor filled with 1.0
    kernel = tf.ones((sigma * 2 + 1,))

    # Calculate the Cauchy kernel values
    x = np.arange(-sigma, sigma + 1)
    kernel = 1 / (1 + (x / sigma) ** 2)

    # Normalize the kernel to ensure its sum equals 1
    kernel = kernel / np.sum(kernel)

    # Return the kernel as a TensorFlow tensor
    return tf.convert_to_tensor(kernel)

if __name__ == "__main__":
    # Create a sample input value
    sigma = 3

    # Call the function and print the result
    kernel = cauchy_kernel1d(sigma)
    print(kernel)

    # Validate the output shape
    assert kernel.shape == (sigma * 2 + 1,)

    # Validate the sum of kernel elements
    assert np.isclose(np.sum(kernel), 1.0)