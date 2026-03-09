import numpy as np
import tensorflow as tf

def gaussian_filter_3d(kernel_sigma):
    if isinstance(kernel_sigma, int):
        kernel_sigma = [kernel_sigma] * 3

    x = np.linspace(-3 * kernel_sigma[0], 3 * kernel_sigma[0], 6 * kernel_sigma[0])
    y = np.linspace(-3 * kernel_sigma[1], 3 * kernel_sigma[1], 6 * kernel_sigma[1])
    z = np.linspace(-3 * kernel_sigma[2], 3 * kernel_sigma[2], 6 * kernel_sigma[2])
    
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    
    gauss = (1 / (np.sqrt(2 * np.pi) ** 3 * np.prod(kernel_sigma))) * \
             np.exp(-(x**2 / (2 * kernel_sigma[0]**2) + 
                       y**2 / (2 * kernel_sigma[1]**2) + 
                       z**2 / (2 * kernel_sigma[2]**2)))

    return tf.convert_to_tensor(gauss, dtype=tf.float32)

if __name__ == "__main__":
    kernel_sigma = [1, 2, 3]
    gaussian_kernel = gaussian_filter_3d(kernel_sigma)
    print(gaussian_kernel.numpy())