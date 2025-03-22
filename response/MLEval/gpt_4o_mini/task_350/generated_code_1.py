import tensorflow as tf

def method():
    # Sample input data (e.g., sequences of integers representing words)
    input_data = tf.constant([[1, 2, 3], [4, 5, 0], [6, 0, 0]])  # Example input data

    # Define the parameters for the embedding layer
    vocab_size = 10  # Size of the vocabulary
    embedding_dim = 4  # Dimension of the embedding space

    # Create an embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)

    # Apply the embedding layer to the input data
    embedded_sequence = embedding_layer(input_data)

    return embedded_sequence

# Call the method and print the output for validation
output = method()
print(output)