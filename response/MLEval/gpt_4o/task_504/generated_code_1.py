import numpy as np
import tensorflow as tf

def method(input_data, vocab_size, embedding_dim):
    # Create an Embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    # Convert input_data to a tensor if it's not already
    input_tensor = tf.constant(input_data)
    
    # Apply the embedding layer to the input data
    output = embedding_layer(input_tensor)
    
    return output

# Example usage for validation
if __name__ == "__main__":
    # Sample input data
    input_data = [1, 2, 3, 4]  # This should be indices of words in the vocabulary
    vocab_size = 10  # Assume we have 10 different words in the vocabulary
    embedding_dim = 4  # Define the size of the embedding vector

    # Call the method and print the output
    embedded_sequence = method(input_data, vocab_size, embedding_dim)
    print(embedded_sequence)