import tensorflow as tf

def method():
    # Create a simple TensorFlow computation graph
    a = tf.constant(5.0, name='a')
    b = tf.constant(3.0, name='b')
    c = tf.add(a, b, name='c')
    
    # Run the graph and compute the result
    with tf.Session() as sess:
        output = sess.run(c)
    
    return output

# Call the method for validation
result = method()
print("Result of the graph computation:", result)