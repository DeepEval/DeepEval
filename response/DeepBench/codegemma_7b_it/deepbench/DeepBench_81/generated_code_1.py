import tensorflow as tf

def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    assert kernel_size >= 3
    assert kernel_size % 2 != 0
    center = int(kernel_size / 2)
    kernel_weights = tf.ones(kernel_size)
    kernel_weights = tf.clip_by_value(kernel_weights, clip_value_min=0.0, clip_value_max=float(center + 1))
    return kernel_weights

if __name__ == "__main__":
    kernel_size = 5
    kernel_weights = triangular_kernel1d(kernel_size)
    print(kernel_weights)