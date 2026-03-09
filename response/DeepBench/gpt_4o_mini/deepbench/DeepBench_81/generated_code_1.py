import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    assert kernel_size >= 3
    assert kernel_size % 2 != 0
    
    half_size = kernel_size // 2
    kernel_weights = tf.range(half_size + 1, dtype=tf.float32)
    kernel_weights = tf.concat([kernel_weights, kernel_weights[:-1][::-1]], axis=0)
    
    return kernel_weights

if __name__ == "__main__":
    kernel_size = 5  # Example kernel size
    kernel = triangular_kernel1d(kernel_size)
    print("Kernel weights:", kernel.numpy())