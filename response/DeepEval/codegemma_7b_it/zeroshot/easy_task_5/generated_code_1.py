import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  # Create the model using Functional API
  inputs = keras.Input(shape=(28, 28, 1))

  # Reduce dimensionality with 1x1 convolution
  conv_1x1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)

  # Extract features using 3x3 convolution
  conv_3x3 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(conv_1x1)

  # Restore dimensionality with 1x1 convolution
  conv_1x1_2 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(conv_3x3)

  # Flatten the output
  flatten = layers.Flatten()(conv_1x1_2)

  # Fully connected layer with 10 neurons for classification
  outputs = layers.Dense(10, activation='softmax')(flatten)

  # Create the model
  model = keras.Model(inputs=inputs, outputs=outputs)

  return model