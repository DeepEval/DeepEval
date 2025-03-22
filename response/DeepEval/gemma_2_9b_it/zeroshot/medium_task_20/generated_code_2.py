import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  inputs = keras.Input(shape=(32, 32, 3))

  # Path 1: 1x1 convolution
  x1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

  # Path 2: 1x1 convolution followed by two 3x3 convolutions
  x2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
  x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
  x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)

  # Path 3: 1x1 convolution followed by a 3x3 convolution
  x3 = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
  x3 = layers.Conv2D(64, (3, 3), activation='relu')(x3)

  # Path 4: Max pooling followed by 1x1 convolution
  x4 = layers.MaxPooling2D((2, 2))(inputs)
  x4 = layers.Conv2D(64, (1, 1), activation='relu')(x4)

  # Concatenate outputs from all paths
  x = layers.concatenate([x1, x2, x3, x4])

  # Flatten and dense layer
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)

  # Output layer with softmax activation
  outputs = layers.Dense(10, activation='softmax')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  return model