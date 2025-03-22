import tensorflow as tf

def method():
    # Define a simple graph with two tensors
    a = tf.constant(2)
    b = tf.constant(3)
    c = tf.add(a, b)

    # Run the graph and retrieve the result
    with tf.compat.v1.Session() as sess:
        output = sess.run(c)

    return output

# Call the method and print the output
result = method()
print("Output:", result)