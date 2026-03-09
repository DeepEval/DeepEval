import tensorflow as tf

def separable_filter(tensor, kernel):
    # Create a 3D separable filter
    filter_depth, filter_height, filter_width = kernel.get_shape()
    in_channels, out_channels = tensor.get_shape()[-1]

    # Initialize the filter weights
    filters = tf.Variable(tf.truncated_normal(shape=(filter_depth, filter_height, filter_width, in_channels, out_channels), stddev=0.1))

    # Create a 3D separable convolutional layer
    conv_layer = tf.nn.conv3d(tensor, filters, strides=[1, 1, 1, 1, 1], padding="SAME")

    # Apply the activation function
    act_layer = tf.nn.relu(conv_layer)

    return act_layer

# Create a sample input tensor
input_tensor = tf.random.normal(shape=(3, 256, 256, 3, 1))

# Create a sample kernel
kernel = tf.random.normal(shape=(3, 3, 3))

# Call the function and print the results
output = separable_filter(input_tensor, kernel)
print(output)

if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = tf.random.normal(shape=(3, 256, 256, 3, 1))

    # Create a sample kernel
    kernel = tf.random.normal(shape=(3, 3, 3))

    # Call the function and print the results
    output = separable_filter(input_tensor, kernel)
    print(output)