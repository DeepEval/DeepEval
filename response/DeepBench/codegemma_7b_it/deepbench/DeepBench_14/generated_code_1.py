import tensorflow as tf

def do_conv2d(conv, input_, padding=None, weight=None, bias=None):
    if weight is None:
        weight = conv.weights[0]
    if bias is None:
        bias = conv.bias
    if padding is None:
        padding = conv.padding

    output = tf.nn.conv2d(input_, weight, strides=conv.strides, padding=padding)

    if bias is not None:
        output += bias

    return output

if __name__ == "__main__":
    # Create sample input values
    input_ = tf.random.normal((1, 28, 28, 3))

    # Create a convolutional layer
    conv = tf.keras.layers.Conv2D(32, (3, 3), padding='same')

    # Perform convolution operation
    output = do_conv2d(conv, input_)

    # Print the results
    print(output.shape)