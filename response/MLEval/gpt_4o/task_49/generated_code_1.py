# Import necessary packages
import tensorflow as tf

def method():
    # Define a simple computation graph
    x = tf.constant(2.0, name='x')
    y = tf.constant(3.0, name='y')
    z = tf.add(x, y, name='z')

    # Create a session to run the graph
    with tf.Session() as sess:
        # Create a summary writer to write summaries to disk
        writer = tf.summary.FileWriter('./logs', sess.graph)
        
        # Run the graph
        output = sess.run(z)
        
        # Close the summary writer
        writer.close()
    
    return output

# Call the method for validation
result = method()
print(f'Result of the computation: {result}')