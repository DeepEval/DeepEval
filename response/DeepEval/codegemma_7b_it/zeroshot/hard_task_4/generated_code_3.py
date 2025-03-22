import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
  # Input layer
  inputs = keras.Input(shape=(32, 32, 3))

  # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
  x = layers.Conv2D(filters=32, kernel_size=1, padding='same')(inputs)

  # Extract initial features using a 3x3 depthwise separable convolution
  x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)

  # Compute channel attention weights through global average pooling followed by two fully connected layers
  attention = layers.GlobalAveragePooling2D()(x)
  attention = layers.Dense(32, activation='relu')(attention)
  attention = layers.Dense(32, activation='sigmoid')(attention)

  # Reshape the attention weights to match the initial features
  attention = layers.Reshape((32, 32, 32))(attention)

  # Multiply the initial features with the attention weights to achieve channel attention weighting
  x = tf.multiply(x, attention)

  # Reduce the dimensionality of the output using a 1x1 convolution and combine it with the initial input
  output = layers.Conv2D(filters=10, kernel_size=1, padding='same')(x)
  output = tf.add(output, inputs)

  # Flatten the output and pass it through a fully connected layer for classification
  output = layers.Flatten()(output)
  output = layers.Dense(10, activation='softmax')(output)

  # Construct the model
  model = keras.Model(inputs=inputs, outputs=output)

  return model