import tensorflow as tf

# Define the input data
input_data = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Define the embedding layer
embedding = tf.keras.layers.Embedding(input_dim=3, output_dim=3)

# Apply the embedding layer to the input data
output = embedding(input_data)

# Print the embedded sequence
print(output)