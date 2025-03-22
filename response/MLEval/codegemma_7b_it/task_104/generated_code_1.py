import tensorflow as tf
import numpy as np

def method():

    # Define the embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=64)

    # Apply embedding to the input data
    embedded_sequence = embedding_layer(input_data)

    # Return the embedded sequence
    return embedded_sequence

# Call the method to validate the code
embedded_sequence = method()