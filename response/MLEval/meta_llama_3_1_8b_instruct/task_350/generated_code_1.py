import tensorflow as tf

def method():
    # Define the input data
    input_data = tf.constant([[1, 2, 3], [4, 5, 6]])  # Input shape: (batch_size, sequence_length)

    # Define the embedding dimension and vocabulary size
    vocab_size = 10
    embedding_dim = 5

    # Apply embedding to the input data
    embedded_sequence = tf.nn.embedding_lookup(tf.Variable(tf.random.normal([vocab_size, embedding_dim])), input_data)

    return embedded_sequence

# Call the method for validation
output = method()
print(output)