import tensorflow as tf

def method():
    # Define the graph
    # Create two placeholder tensors for input
    a = tf.placeholder(tf.float32, shape=[None, 3])
    b = tf.placeholder(tf.float32, shape=[3, None])
    
    # Define the operation (matrix multiplication in this case)
    c = tf.matmul(a, b)
    
    # Create a session to run the graph
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        
        # Define some input data
        input_a = [[1, 2, 3]]
        input_b = [[4], [5], [6]]
        
        # Run the graph and get the output
        output = sess.run(c, feed_dict={a: input_a, b: input_b})
    
    return output

# Call the method to validate
print(method())