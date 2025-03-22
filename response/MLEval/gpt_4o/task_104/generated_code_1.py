import tensorflow as tf
import numpy as np

def method(input_data, vocab_size, embedding_dim):
    # Define the Embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # Apply the embedding to the input_data
    embedded_sequences = embedding_layer(input_data)

    return embedded_sequences

# Example usage for validation
# Define input parameters
input_data = np.array([[1, 2, 3], [4, 5, 6]])  # Example input data (e.g., tokenized word indices)
vocab_size = 10  # Example vocabulary size
embedding_dim = 8  # Example embedding dimension

# Call the method and print the output
embedded_output = method(input_data, vocab_size, embedding_dim)
print(embedded_output)