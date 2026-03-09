import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D triangular kernel for LocalNormalizedCrossCorrelation.
    Assume kernel_size is odd, it will be a smoothed from a kernel which center part is zero.
    Then length of the ones will be around half kernel_size.
    The weight scale of the kernel does not matter as LNCC will normalize it.
    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    assert kernel_size >= 3, "kernel_size must be greater than or equal to 3"
    assert kernel_size % 2 != 0, "kernel_size must be odd"

    # Generate a kernel with the given size
    kernel_weights = tf.zeros(shape=(kernel_size,))

    # Set the middle value to 1
    kernel_weights[kernel_size // 2] = 1

    # Smooth the kernel with a Gaussian filter
    kernel_weights = tf.nn.gaussian_filter(
        kernel_weights, sigma=kernel_size // 2, mode="nearest"
    )

    return kernel_weights

# Example usage
if __name__ == "__main__":
    # Generate a sample input tensor
    input_tensor = tf.random.normal(shape=(1, 10, 10))

    # Apply the triangular kernel to the input tensor
    output_tensor = triangular_kernel1d(kernel_size=5)(input_tensor)

    # Print the output tensor
    print(output_tensor)