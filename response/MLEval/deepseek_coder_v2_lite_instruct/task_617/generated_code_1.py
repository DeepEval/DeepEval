import tensorflow as tf

def method():
    # Define the input data (example placeholder)
    input_data = tf.placeholder(tf.int32, shape=[None, None])
    
    # Define the embedding layer
    embedding_dim = 10  # Example dimension for the embedding
    embedding_layer = tf.get_variable("embedding_layer", [vocab_size, embedding_dim])
    
    # Apply the embedding layer to the input data
    embedded_sequence = tf.nn.embedding_lookup(embedding_layer, input_data)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Create a session
    with tf.Session() as sess:
        sess.run(init)
        # Example input data
        example_input = [[1, 2, 3], [4, 5, 6]]
        output = sess.run(embedded_sequence, feed_dict={input_data: example_input})
    
    return output

# Call the method for validation
output = method()
print(output)