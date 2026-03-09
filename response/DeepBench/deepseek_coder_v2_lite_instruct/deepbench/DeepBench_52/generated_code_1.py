import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    # Apply depthwise convolution
    depthwise_conv = tf.nn.depthwise_conv3d(tensor, kernel[None, :, None, None, :], strides=[1, 1, 1, 1, 1], padding='SAME')
    
    # Apply pointwise convolution
    pointwise_conv = tf.nn.conv3d(depthwise_conv, kernel[None, None, None, :, None], strides=[1, 1, 1, 1, 1], padding='SAME')
    
    return pointwise_conv

if __name__ == "__main__":
    # Create sample input values
    tensor = tf.random.normal(shape=(1, 32, 32, 32, 1))
    kernel = tf.random.normal(shape=(3,))
    
    # Call the function
    result = separable_filter(tensor, kernel)
    
    # Print the results
    print(result)