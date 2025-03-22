import tensorflow as tf
from tensorflow import keras
from keras import layers

def dl_model():
  inputs = keras.Input(shape=(32, 32, 3))

  x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
  x = layers.concatenate([inputs, x])
  x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
  x = layers.concatenate([inputs, x])
  x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)

  x = layers.Flatten()(x)
  outputs = layers.Dense(10, activation="softmax")(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  return model