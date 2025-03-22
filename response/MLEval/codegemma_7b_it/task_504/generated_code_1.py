import tensorflow as tf

def method(input_data):
    # Define the embedding layer
    embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=128)

    # Apply the embedding to the input data
    embedded_data = embedding(input_data)

    # Return the embedded sequence
    return embedded_data

# Validation code
input_data = tf.keras.preprocessing.sequence.pad_sequences([[1, 2, 3], [4, 5, 6], [7, 8, 9]], maxlen=10)
embedded_data = method(input_data)

print(embedded_data)