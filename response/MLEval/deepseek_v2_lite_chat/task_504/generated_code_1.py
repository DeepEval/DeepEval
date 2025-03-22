import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

# Ensure reproducibility
tf.random.set_seed(123)

# Define the method
def method(input_data, vocab_size, embedding_dim):
    # Input data should be a 2D tensor where the first dimension is the batch size
    # and the second dimension is the sequence length.
    # Example: input_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # Pad the sequences to a fixed length (for example, 100)
    max_len = 100
    input_data = sequence.pad_sequences(input_data, maxlen=max_len)
    
    # Create an embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(input_data)
    
    # Return the embedded sequence
    return embedding_layer

# Example usage
vocab_size = 1000  # Replace with your actual vocabulary size
embedding_dim = 50  # Replace with the desired dimension of the embedding

# Generate some example input data
input_data = [[3, 5, 2], [7, 10, 1], [1, 4, 6]]

# Call the method
output = method(input_data, vocab_size, embedding_dim)

# Print the output
print("Output:", output)