import tensorflow as tf

def method():
  # Define a sample input tensor
  input_tensor = tf.random.normal(shape=(1, 10))

  # Define a sample fully connected layer
  dense_layer = tf.keras.layers.Dense(units=3, activation='linear')(input_tensor)

  # Apply softmax activation
  output = tf.nn.softmax(dense_layer)

  return output

# Call the method and print the output
output = method()
print(output)