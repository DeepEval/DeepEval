import tensorflow as tf

def method():
    # Create a simple TensorFlow session.
    sess = tf.Session()

    # Define a constant tensor.
    x = tf.constant(10)

    # Print the tensor value.
    output = sess.run(x)

    # Print the output.
    print(output)

    return output

# Call the method.
method()