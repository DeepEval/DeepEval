import tensorflow as tf

def method():
  # Input data
  input_data = tf.constant(
      [
          "the quick brown fox jumps over the lazy dog",
          "the cat sat on the mat",
      ],
      dtype=tf.string,
  )

  # Create an embedding layer
  embedding_layer = tf.keras.layers.Embedding(
      input_dim=10000,  # Vocabulary size
      output_dim=16,  # Embedding dimension
  )

  # Apply embedding to input data
  embedded_sequences = embedding_layer(input_data)

  # Print the embedded sequences
  print(embedded_sequences)

  return embedded_sequences

# Call the method
method()