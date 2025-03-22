import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  # Define the input layer
  input_layer = layers.Input(shape=(32, 32, 3))

  # Compress the input features with global average pooling
  x = layers.GlobalAveragePooling2D()(input_layer)

  # Add two fully connected layers to generate weights
  x = layers.Dense(32)(x)
  weights = layers.Dense(3)(x)

  # Reshape the weights to match the input shape
  weights = layers.Reshape((32, 3))(weights)

  # Multiply the input feature map with the weights
  weighted_map = layers.multiply([input_layer, weights])

  # Flatten the weighted feature map
  x = layers.Flatten()(weighted_map)

  # Add the final fully connected layer for output
  output_layer = layers.Dense(10, activation='softmax')(x)

  # Construct the model
  model = keras.Model(inputs=input_layer, outputs=output_layer)

  return model