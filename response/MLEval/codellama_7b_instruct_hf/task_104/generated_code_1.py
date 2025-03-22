import tensorflow as tf

# Define the input data
input_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Define the embedding layer
embedding = tf.keras.layers.Embedding(input_dim=10, output_dim=10, input_length=3)

# Convert the input data to a tensor
input_tensor = tf.convert_to_tensor(input_data, dtype=tf.int32)

# Embed the input data
embedded_data = embedding(input_tensor)

# Print the embedded data
print(embedded_data)