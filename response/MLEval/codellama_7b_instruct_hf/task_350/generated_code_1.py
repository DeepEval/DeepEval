import tensorflow as tf

# Define the input data
input_data = [["this", "is", "a", "test"], ["this", "is", "another", "test"]]

# Create the embedding layer
embedding_layer = tf.keras.layers.Embedding(input_dim=100, output_dim=128, input_length=4)

# Apply the embedding layer to the input data
output = embedding_layer(input_data)

# Print the output
print(output)