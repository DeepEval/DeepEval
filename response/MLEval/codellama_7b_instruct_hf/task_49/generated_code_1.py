import tensorflow as tf

def method():
    # Create a session
    sess = tf.Session()

    # Define a variable
    x = tf.Variable(1.0, name='x')

    # Define a summary writer
    writer = tf.summary.FileWriter('logs/', sess.graph)

    # Add the variable to the summary writer
    writer.add_summary(tf.summary.scalar('x', x))

    # Close the session
    sess.close()

    # Return the output
    return output

# Call the method for validation
output = method()