import tensorflow as tf

def method():
    # Create a simple computation graph
    a = tf.constant(3, name='constant_a')
    b = tf.constant(4, name='constant_b')
    c = tf.add(a, b, name='add_op')
    
    # Create a summary writer to log data for TensorBoard
    logdir = "logs/"
    summary_writer = tf.summary.create_file_writer(logdir)

    # Log the operation names
    with summary_writer.as_default():
        tf.summary.scalar('a', a.numpy(), step=0)
        tf.summary.scalar('b', b.numpy(), step=0)
        tf.summary.scalar('c', c.numpy(), step=0)
    
    # Flush the writer
    summary_writer.flush()
    
    # Return the output value for validation
    output = c.numpy()
    return output

# Call the method for validation
result = method()
print("Output:", result)