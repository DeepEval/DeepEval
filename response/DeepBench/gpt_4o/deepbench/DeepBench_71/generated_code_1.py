import tensorflow as tf
import numpy as np

def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    assert sigma > 0
    
    # Define the kernel size; a common choice is 6 times the sigma
    kernel_size = 6 * sigma
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd
    
    # Create a 1D range centered around zero
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=np.float32)
    
    # Calculate the Cauchy kernel
    kernel = 1.0 / (1.0 + (x / sigma) ** 2)
    
    # Normalize the kernel
    kernel /= tf.reduce_sum(kernel)
    
    return tf.convert_to_tensor(kernel, dtype=tf.float32)

if __name__ == "__main__":
    # Sample input value for sigma
    sigma = 2
    # Call the function
    kernel = cauchy_kernel1d(sigma)
    # Print the result
    print("Cauchy Kernel 1D for sigma =", sigma, ":\n", kernel.numpy())