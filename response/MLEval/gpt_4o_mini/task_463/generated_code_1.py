import tensorflow as tf
import numpy as np

def method():
    # Sample input data: integer indices representing words
    input_data = np.array([[1, 2, 3], [4, 5, 6]])  # Example input with shape (2, 3)
    
    # Define the parameters for the embedding layer
    vocab_size = 10  # Example vocabulary size
    embedding_dim = 4  # Dimension of the embedding

    # Create the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # Apply the embedding layer to the input data
    embedded_output = embedding_layer(input_data)

    return embedded_output

# Call the method for validation
output = method()
print(output)