import tensorflow as tf

def method():
    # Create a summary writer
    writer = tf.summary.create_file_writer('logs')
    
    # Example: Log a simple scalar value
    with writer.as_default():
        tf.summary.scalar('my_metric', 0.7, step=0)
        tf.summary.scalar('my_metric', 0.8, step=1)
        tf.summary.scalar('my_metric', 0.9, step=2)
    
    # Close the writer
    writer.close()
    
    # Return the output (if any)
    output = "TensorBoard logs have been written to 'logs' directory."
    return output

# Call the method for validation
output = method()
print(output)