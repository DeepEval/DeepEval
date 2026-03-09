import numpy as np
import tensorflow as tf

def gaussian_filter_3d(kernel_sigma):
    if isinstance(kernel_sigma, int):
        kernel_sigma = [kernel_sigma, kernel_sigma, kernel_sigma]
    elif len(kernel_sigma)!= 3:
        raise ValueError("kernel_sigma must be an integer or a list of three integers")
    
    # Calculate the standard deviations for the Gaussian distribution
    std_devs = [2 * sig ** 2 for sig in kernel_sigma]
    
    # Create a 1D Gaussian kernel for each dimension
    kernel_1d = tf.exp(-((np.linspace(-3, 3, 7) / std_devs[0]) ** 2) / 2)
    kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
    
    # Create the 3D Gaussian kernel by taking the outer product of the 1D kernels
    kernel_3d = tf.tensordot(tf.tensordot(kernel_1d, kernel_1d, 0), kernel_1d, 0)
    
    return kernel_3d

if __name__ == "__main__":
    # Create sample input values
    kernel_sigma_list = [1, 1, 1]
    kernel_sigma_int = 1
    
    # Call the function and print the results
    kernel_list = gaussian_filter_3d(kernel_sigma_list)
    kernel_int = gaussian_filter_3d(kernel_sigma_int)
    
    print("Gaussian kernel for list of sigma values:")
    print(kernel_list)
    
    print("\nGaussian kernel for single sigma value:")
    print(kernel_int)