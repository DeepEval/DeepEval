from tensorflow.keras.layers import Reshape, Permute, Input, Dense, Activation
from tensorflow.keras.models import Model

def dl_model():
  # Input layer
  inputs = Input(shape=(None, None, 3))

  # Reshape and channel shuffling
  x = Reshape((None, None, 3, 1))(inputs)  # Reshape to (height, width, groups, channels_per_group)
  x = Permute((0, 1, 3, 2))(x)  # Swap third and fourth dimensions for channel shuffling

  # Reshape back to original input shape
  x = Reshape((None, None, 3))(x)

  # Fully connected layer for classification
  x = Dense(10, activation='softmax')(x)

  # Create model
  model = Model(inputs=inputs, outputs=x)

  return model