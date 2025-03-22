import tensorflow as tf

def method():
    # Example input data (batch of sequences)
    input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    
    # Define the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=10, output_dim=5, input_length=3)
    
    # Apply the embedding layer to the input data
    embedded_sequence = embedding_layer(input_data)
    
    # Convert the embedded sequence to a numpy array (for easier validation)
    output = embedded_sequence.numpy()
    
    return output

# Call the method for validation
output = method()
print(output)