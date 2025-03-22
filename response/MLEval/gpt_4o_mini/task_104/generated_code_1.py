import numpy as np
import tensorflow as tf

def method(input_data, vocab_size, embedding_dim):
    # Create an Embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    # Convert input_data to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.int32)
    
    # Apply the embedding layer
    embedded_sequence = embedding_layer(input_tensor)
    
    return embedded_sequence.numpy()  # Convert the output to a NumPy array for easier handling

# Example usage:
if __name__ == "__main__":
    # Sample input data (e.g., tokenized sentences)
    input_data = [[1, 2, 3], [4, 5, 6]]  # Example tokenized sequences
    vocab_size = 10  # Size of the vocabulary
    embedding_dim = 4  # Dimensionality of the embedding space

    output = method(input_data, vocab_size, embedding_dim)
    print(output)