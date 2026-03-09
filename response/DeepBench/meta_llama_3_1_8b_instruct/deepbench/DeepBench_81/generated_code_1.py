import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    ones_size = (kernel_size + 1) // 2
    kernel_weights = tf.ones((ones_size,)) * (kernel_size - ones_size)
    center = tf.zeros((kernel_size - ones_size,))
    return tf.concat([kernel_weights, center, kernel_weights], axis=0)

if __name__ == "__main__":
    kernel_size = 7
    kernel = triangular_kernel1d(kernel_size)
    print(kernel)
    print(kernel.shape)