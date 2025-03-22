import keras
from keras import layers

def dl_model():
  # Input layer
  inputs = keras.Input(shape=(28, 28, 1))

  # Block 1
  block1 = layers.MaxPooling2D(pool_size=1, strides=1, padding="valid")(inputs)
  block1 = layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(block1)
  block1 = layers.MaxPooling2D(pool_size=4, strides=4, padding="valid")(block1)
  block1 = layers.Flatten()(block1)

  # Reshape block 1 output
  block1 = layers.Reshape((4, 4, 1))(block1)

  # Block 2
  branch1 = layers.Conv2D(filters=10, kernel_size=1, padding="same", activation="relu")(block1)
  branch2 = layers.Conv2D(filters=10, kernel_size=3, padding="same", activation="relu")(block1)
  branch3 = layers.Conv2D(filters=10, kernel_size=5, padding="same", activation="relu")(block1)
  branch4 = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(block1)

  # Concatenate branches
  concat = layers.concatenate([branch1, branch2, branch3, branch4])

  # Classification layer
  outputs = layers.Flatten()(concat)
  outputs = layers.Dense(units=10, activation="softmax")(outputs)

  # Model creation
  model = keras.Model(inputs=inputs, outputs=outputs)

  return model