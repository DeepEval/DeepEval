# Import necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Embedding

# Define input data and vocabulary size
input_data = [1, 2, 3, 4, 5]
vocab_size = 10
embedding_dim = 5

# Define method to apply embedding to input data
def method():
    """
    Apply embedding to `input_data` using TensorFlow.
    
    Returns:
        embedded_sequence (tf.Tensor): The embedded sequence.
    """
    
    # Create an Embedding layer
    embedding_layer = Embedding(vocab_size, embedding_dim)
    
    # Convert input data to a tensor
    input_tensor = tf.convert_to_tensor(input_data)
    
    # Apply embedding to the input tensor
    embedded_sequence = embedding_layer(input_tensor)
    
    # Return the embedded sequence
    return embedded_sequence

# Call the method for validation
output = method()
print(output)