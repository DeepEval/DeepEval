import tensorflow as tf

def method():
    # Define some logits (raw, unnormalized scores)
    logits = tf.constant([1.0, 2.0, 3.0])
    
    # Apply the softmax function
    softmax_output = tf.nn.softmax(logits)
    
    # Initialize and run the session to get the result
    with tf.Session() as sess:
        output = sess.run(softmax_output)
    
    return output

# Call the method and print the result for validation
output = method()
print(output)