import tensorflow as tf
from tensorflow.keras.layers import Embedding

def method():
    # Example input data (batch of sequences)
    input_data = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Define the embedding layer
    embedding_layer = Embedding(input_dim=10, output_dim=5, input_length=3)
    
    # Apply the embedding layer to input_data
    embedded_sequence = embedding_layer(input_data)
    
    # Return the embedded sequence
    return embedded_sequence

# Call the method for validation
output = method()
print(output)