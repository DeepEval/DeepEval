import tensorflow as tf

def gaussian_filter_3d(kernel_sigma):
    """
    Defines a 3D Gaussian filter for smoothing.

    Args:
        kernel_sigma: A list of deviations for each direction or an integer for isotropic deviation.

    Returns:
        A TensorFlow tensor representing a Gaussian kernel of shape [3*k for k in kernel_sigma].
    """
    if isinstance(kernel_sigma, int):
        kernel_sigma = [kernel_sigma, kernel_sigma, kernel_sigma]

    kernel_size = [3*k for k in kernel_sigma]
    kernel = tf.constant(np.random.normal(size=kernel_size))
    kernel = kernel / kernel.sum()

    return kernel

if __name__ == "__main__":
    # Create sample input values
    input_values = np.random.normal(size=(10, 10, 10))

   # Call the function and print the results
    kernel = gaussian_filter_3d(kernel_sigma=3)
    output = tf.nn.conv3d(input_values, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
    print(output)