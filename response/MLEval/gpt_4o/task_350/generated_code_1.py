import tensorflow as tf
import numpy as np

def method():
    # Sample input data: a batch of sequences (e.g., sentences represented as sequences of word indices)
    input_data = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    
    # Define parameters for the embedding layer
    vocab_size = 13  # Assume the vocabulary size is 13 (including a padding or OOV token)
    embedding_dim = 4  # Dimension of the embedding space

    # Create an embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # Get the embedded output
    output = embedding_layer(input_data)

    return output

# Call the method for validation
embedded_sequence = method()
print(embedded_sequence)