import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding

def method(input_data, output_dim=10, input_length=5, embeddings_initializer='uniform', trainable=False):
    # Input data should be a list of integers
    # Each integer is assumed to be the index of a word in a vocabulary
    
    # Create an Embedding layer
    embedding_layer = Embedding(input_dim=input_data.shape[0],
                                output_dim=output_dim,
                                input_length=input_length,
                                embeddings_initializer=embeddings_initializer,
                                trainable=trainable)
    
    # Expand dimensions to allow for batch processing (optional)
    input_data = np.expand_dims(input_data, axis=0)
    
    # Convert input data to a tensor
    input_tensor = tf.convert_to_tensor(input_data)
    
    # Apply the embedding layer
    embedded_data = embedding_layer(input_tensor)
    
    # Remove the batch dimension if it exists
    if embedded_data.shape[0].value == 1:
        embedded_data = embedded_data[0]
    
    return embedded_data

# Example usage
vocab_size = 20000  # Example vocabulary size
num_words = 10000    # Example number of words to consider (for simplicity)
embedding_dim = 10   # Embedding dimension size

# Example input data
input_data = np.random.randint(0, num_words, size=(100, input_length))

# Call the method with the example input data
output = method(input_data, output_dim=embedding_dim, input_length=input_length)

print("Output:", output)