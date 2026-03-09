import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    # Create separable filters
    filter_depth = tf.reshape(kernel, (len(kernel), 1, 1, 1, 1))
    filter_height = tf.reshape(kernel, (1, len(kernel), 1, 1, 1))
    filter_width = tf.reshape(kernel, (1, 1, len(kernel), 1, 1))
    
    # Apply 3D convolution for each dimension separately
    conv_depth = tf.nn.conv3d(tensor, filter_depth, strides=[1, 1, 1, 1, 1], padding='SAME')
    conv_height = tf.nn.conv3d(conv_depth, filter_height, strides=[1, 1, 1, 1, 1], padding='SAME')
    conv_width = tf.nn.conv3d(conv_height, filter_width, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    return conv_width

if __name__ == "__main__":
    # Create a sample tensor of shape (batch, dim1, dim2, dim3, 1)
    sample_tensor = tf.constant([[[[[1], [2]], [[3], [4]]],
                                  [[[5], [6]], [[7], [8]]]]], dtype=tf.float32)

    # Create a sample kernel
    sample_kernel = tf.constant([1, 2, 1], dtype=tf.float32)

    # Apply separable filter
    output = separable_filter(sample_tensor, sample_kernel)

    # Print the output tensor
    print(output.numpy())