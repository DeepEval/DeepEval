import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    # Reshape the kernel to (1, 1, dim4, 1, 1)
    kernel = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(kernel, 2), 2), 3), 4)
    # Reshape the tensor to (batch, 1, dim1, dim2, dim3, 1)
    tensor = tf.expand_dims(tensor, 1)
    # Apply the filter
    output = tf.nn.conv3d(tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    # Reshape the output back to (batch, dim1, dim2, dim3, 1)
    output = output[:, 0, :, :, :, 0]
    return output

if __name__ == "__main__":
    # Create sample input values
    batch = 2
    dim1 = 5
    dim2 = 5
    dim3 = 5
    kernel_dim4 = 3
    tensor = tf.random.uniform((batch, dim1, dim2, dim3, 1))
    kernel = tf.random.uniform((kernel_dim4,))
    
    # Call the function
    output = separable_filter(tensor, kernel)
    
    # Print the results
    print("Input Tensor Shape:", tensor.shape)
    print("Kernel Shape:", kernel.shape)
    print("Output Shape:", output.shape)
    print("Output Tensor Shape:", tensor.shape)