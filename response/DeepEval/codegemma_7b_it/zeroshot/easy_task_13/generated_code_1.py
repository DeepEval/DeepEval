import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  inputs = keras.Input(shape=(28, 28, 1))

  # First set of convolutional layers
  x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.2)(x)

  # Second set of convolutional layers
  x = layers.Conv2D(filters=64, kernel_size=(3, 1), padding='same')(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(filters=64, kernel_size=(1, 3), padding='same')(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.2)(x)

  # Third set of convolutional layers
  x = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.2)(x)

  # Combine processed features with original input
  x = layers.Add()([x, inputs])

  # Flatten and fully connected layer
  x = layers.Flatten()(x)
  outputs = layers.Dense(units=10, activation='softmax')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model