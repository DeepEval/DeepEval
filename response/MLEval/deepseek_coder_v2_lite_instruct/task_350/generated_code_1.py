import tensorflow as tf
from tensorflow.keras.layers import Embedding

def method():
    # Define the input data (example sequence)
    input_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # Define the embedding layer
    embedding_layer = Embedding(input_dim=10, output_dim=5, input_length=3)
    
    # Convert input_data to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.int32)
    
    # Apply the embedding layer
    embedded_sequence = embedding_layer(input_tensor)
    
    # Convert the embedded sequence back to a Python list for the return value
    output = embedded_sequence.numpy()
    
    return output

# Call the method for validation
output = method()
print(output)