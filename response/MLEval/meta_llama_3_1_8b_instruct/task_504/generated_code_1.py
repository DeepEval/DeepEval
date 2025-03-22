# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method():
    """
    Apply embedding to input data using TensorFlow and return the embedded sequence.
    
    Returns:
    embedded_sequence (tf.Tensor): The embedded sequence.
    """

    # Define the input data
    input_data = tf.constant([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # Define the embedding layer
    embedding_layer = layers.Embedding(
        input_dim=10,  # Maximum possible input word index + 1
        output_dim=5,  # Dimensionality of the embedding vector
        input_length=tf.shape(input_data)[1]  # Sequence length
    )

    # Apply the embedding layer to the input data
    embedded_sequence = embedding_layer(input_data)

    return embedded_sequence

# Call the method for validation
output = method()
print(output)