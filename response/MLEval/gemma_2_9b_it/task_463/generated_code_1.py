import tensorflow as tf

def method(input_data):
  # Define the embedding layer
  embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=128)  

  # Apply the embedding layer to the input data
  output = embedding_layer(input_data)

  return output

# Example usage:
input_data = tf.constant([[1, 2, 3], [4, 5, 6]])  # Example input sequence
embedded_sequences = method(input_data)

print(embedded_sequences)