import tensorflow as tf
from tensorflow.keras import layers

# Define the input data
input_data = tf.random.uniform(shape=(1, 10), minval=0, maxval=10, dtype=tf.int32)

# Define the embedding dimension and vocabulary size
embedding_dim = 10
vocab_size = 10

# Create a method to apply embedding to the input data
def method():
    # Create an embedding layer with the specified vocabulary size and dimension
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    # Apply the embedding layer to the input data
    embedded_sequence = embedding(input_data)
    
    return embedded_sequence

# Call the method for validation
output = method()
print(output.shape)