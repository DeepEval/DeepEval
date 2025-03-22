import tensorflow as tf
from tensorflow.keras.layers import Embedding

def method():
    # Define the embedding layer
    embedding_layer = Embedding(vocab_size, embedding_size)

    # Embed the input data
    embedded_sequence = embedding_layer(input_data)

    return embedded_sequence

# Call the method for validation
embedded_sequence = method()

print(embedded_sequence)