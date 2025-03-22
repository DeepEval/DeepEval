import tensorflow as tf

def method():
    # Create a simple TensorFlow graph
    a = tf.constant(2)
    b = tf.constant(3)
    c = a + b

    # Write the graph to a log directory
    writer = tf.summary.create_file_writer('./logs')
    with writer.as_default():
        tf.summary.scalar('c', c, step=0)

    # Optionally, read and print the output
    output = "TensorFlow graph has been written to ./logs"
    return output

# Call the method to validate the output
print(method())