import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """ Return a the 1D triangular kernel for LocalNormalizedCrossCorrelation. Assume kernel_size is odd, it will be a smoothed from a kernel which center part is zero. Then length of the ones will be around half kernel_size. The weight scale of the kernel does not matter as LNCC will normalize it. :param kernel_size: scalar, size of the 1-D kernel :return: kernel_weights, of shape (kernel_size, ) """
    assert kernel_size >= 3, "Kernel size must be at least 3"
    assert kernel_size % 2 != 0, "Kernel size must be odd"

    # Create a triangular kernel with ones in the middle and zeros on both sides
    kernel_weights = tf.ones(kernel_size)
    kernel_weights[kernel_size // 2] = 0.5  # Set the middle value to 0.5

    return kernel_weights

if __name__ == "__main__":
    # Create a sample input value
    kernel_size = 7

    # Call the function and print the result
    kernel_weights = triangular_kernel1d(kernel_size)
    print("Triangular kernel weights for kernel size", kernel_size, "are:", kernel_weights.numpy())