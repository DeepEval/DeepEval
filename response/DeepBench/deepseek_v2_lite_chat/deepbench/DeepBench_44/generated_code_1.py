import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def gaussian_filter_3d(kernel_sigma):
    if isinstance(kernel_sigma, list):
        kernel_sigma = np.array(kernel_sigma)
        kernel_sigma = kernel_sigma.reshape((3, len(kernel_sigma)))
    elif isinstance(kernel_sigma, int):
        kernel_sigma = np.full(3, kernel_sigma)
    else:
        raise ValueError("kernel_sigma must be a list of deviations for each direction or an integer for isotropic deviation.")
    
    # Create a Gaussian kernel tensor
    sigma = kernel_sigma
    x = np.linspace(-(sigma[1] // 2), (sigma[1] // 2), sigma[1])
    y = np.linspace(-(sigma[0] // 2), (sigma[0] // 2), sigma[0])
    z = np.linspace(-(sigma[2] // 2), (sigma[2] // 2), sigma[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    K = np.exp(-0.5 * ((X / sigma[0])**2 + (Y / sigma[1])**2 + (Z / sigma[2])**2))
    K /= K.sum()  # Normalize to ensure the kernel sums to 1
    
    # Convert to TensorFlow tensor
    K = tf.constant(K, dtype=tf.float32)
    return K

if __name__ == "__main__":
    # Sample input values
    kernel_sigma = [1, 2, 3]  # List of deviations for each direction
    kernel = gaussian_filter_3d(kernel_sigma)
    
    # Print the results
    print("Gaussian Kernel Tensor:\n", kernel)