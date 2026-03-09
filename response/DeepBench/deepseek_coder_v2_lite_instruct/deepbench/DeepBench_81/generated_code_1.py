import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    # Calculate the midpoint of the kernel
    midpoint = kernel_size // 2
    
    # Create the triangular kernel
    kernel_weights = tf.concat([
        tf.linspace(0., 1., midpoint + 1),
        tf.linspace(1., 0., midpoint)
    ], axis=0)[:kernel_size]
    
    return kernel_weights

if __name__ == "__main__":
    kernel_size = 5
    kernel_weights = triangular_kernel1d(kernel_size)
    print(kernel_weights.numpy())