import tensorflow as tf
from tensorflow.keras import layers

def method():
    # Load input data
    input_data = tf.random.uniform((100, 10))

    # Create an embedding layer
    embedding_layer = layers.Embedding(input_dim=10, output_dim=5)

    # Apply embedding to input data
    embedded_sequences = embedding_layer(input_data)

    # Return the embedded sequence
    return embedded_sequences

# Call the method and print the output
output = method()
print(output.shape)