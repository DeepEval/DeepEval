import tensorflow as tf
import numpy as np

def gaussian_filter_3d(kernel_sigma):
    if isinstance(kernel_sigma, int):
        kernel_sigma = [kernel_sigma] * 3

    kernel_size = [3 * k for k in kernel_sigma]
    kernel_center = [(size - 1) / 2 for size in kernel_size]

    # Generate a grid of (x, y, z) coordinates
    x = np.arange(0, kernel_size[0], 1, float)
    y = np.arange(0, kernel_size[1], 1, float)
    z = np.arange(0, kernel_size[2], 1, float)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate the 3D Gaussian function
    gaussian_kernel = np.exp(
        -(((x - kernel_center[0])**2 / (2 * kernel_sigma[0]**2)) +
          ((y - kernel_center[1])**2 / (2 * kernel_sigma[1]**2)) +
          ((z - kernel_center[2])**2 / (2 * kernel_sigma[2]**2)))
    )

    # Normalize the kernel so that its sum is 1
    gaussian_kernel /= np.sum(gaussian_kernel)

    # Convert to a TensorFlow tensor
    gaussian_kernel_tensor = tf.convert_to_tensor(gaussian_kernel, dtype=tf.float32)

    return gaussian_kernel_tensor

if __name__ == "__main__":
    # Example usage
    kernel_sigma = 1
    gaussian_kernel_tensor = gaussian_filter_3d(kernel_sigma)
    print("Gaussian Kernel for kernel_sigma = 1:")
    print(gaussian_kernel_tensor.numpy())

    kernel_sigma = [1, 2, 3]
    gaussian_kernel_tensor = gaussian_filter_3d(kernel_sigma)
    print("\nGaussian Kernel for kernel_sigma = [1, 2, 3]:")
    print(gaussian_kernel_tensor.numpy())