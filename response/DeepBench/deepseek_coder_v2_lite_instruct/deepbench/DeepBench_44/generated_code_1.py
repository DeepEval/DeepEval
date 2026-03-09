import tensorflow as tf
import numpy as np

def gaussian_filter_3d(kernel_sigma):
    if isinstance(kernel_sigma, int) or isinstance(kernel_sigma, float):
        kernel_sigma = [kernel_sigma, kernel_sigma, kernel_sigma]
    else:
        kernel_sigma = list(kernel_sigma)
    
    # Define the size of the kernel
    kernel_size = [3 * k for k in kernel_sigma]
    
    # Create coordinate grid
    x = tf.range(-kernel_size[0] // 2 + 1, kernel_size[0] // 2 + 1)
    y = tf.range(-kernel_size[1] // 2 + 1, kernel_size[1] // 2 + 1)
    z = tf.range(-kernel_size[2] // 2 + 1, kernel_size[2] // 2 + 1)
    x, y, z = tf.meshgrid(x, y, z, indexing='ij')
    
    # Compute Gaussian values
    gaussian_kernel = tf.exp(-(x**2 + y**2 + z**2) / (2 * kernel_sigma[0]**2))
    gaussian_kernel /= tf.reduce_sum(gaussian_kernel)
    
    return gaussian_kernel

if __name__ == "__main__":
    # Example usage
    kernel_sigma = [1, 2, 3]
    kernel = gaussian_filter_3d(kernel_sigma)
    print(kernel.numpy())