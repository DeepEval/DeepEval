import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    kernel_depth = kernel.shape[0]
    kernel = tf.reshape(kernel, (kernel_depth, 1, 1, 1, 1))  # Reshape to (filter_depth, 1, 1, in_channels, out_channels)

    filtered_tensor = tf.nn.conv3d(tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    return filtered_tensor

if __name__ == "__main__":
    tensor = tf.constant([[[[[1.0]], [[2.0]], [[3.0]]],
                           [[[4.0]], [[5.0]], [[6.0]]],
                           [[[7.0]], [[8.0]], [[9.0]]]]], dtype=tf.float32)  # shape (1, 3, 3, 3, 1)
    
    kernel = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)  # shape (3,)
    
    result = separable_filter(tensor, kernel)
    
    print(result.numpy())