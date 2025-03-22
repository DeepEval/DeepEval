import tensorflow as tf

def method():
  # Define the input layer (assuming 10 input features)
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,))  # Hidden layer with 16 nodes and ReLU activation
  ])

  return model

# Call the method to create the model
my_model = method()

# Print a summary of the model architecture
my_model.summary()