import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
  # Input layer
  input_img = layers.Input(shape=(28, 28, 1))

  # 1x1 convolutional layer to increase dimensionality
  x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same")(input_img)
  x = layers.Activation("relu")(x)

  # 3x3 depthwise separable convolutional layer for feature extraction
  x = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
  x = layers.Activation("relu")(x)

  # 1x1 convolutional layer to reduce dimensionality
  x = layers.Conv2D(filters=10, kernel_size=(1, 1), padding="same", strides=2)(x)
  x = layers.Activation("relu")(x)

  # Flattening layer
  x = layers.Flatten()(x)

  # Fully connected layer for classification
  output = layers.Dense(10, activation="softmax")(x)

  # Model definition
  model = Model(inputs=input_img, outputs=output)

  return model

model = dl_model()
print(model.summary())