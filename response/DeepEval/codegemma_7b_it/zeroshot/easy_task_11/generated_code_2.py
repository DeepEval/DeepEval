import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

  # Load MNIST dataset
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # Normalize pixel values to the range [0, 1]
  x_train = x_train.astype("float32") / 255.0
  x_test = x_test.astype("float32") / 255.0

  # Reshape input data for convolutional layers
  x_train = tf.expand_dims(x_train, -1)
  x_test = tf.expand_dims(x_test, -1)

  # Create model using Functional API
  inputs = keras.Input(shape=(28, 28, 1))

  # Average pooling layer
  x = layers.AveragePooling2D(pool_size=5, strides=3, padding="valid")(inputs)

  # 1x1 convolutional layer
  x = layers.Conv2D(filters=16, kernel_size=1, strides=1, padding="valid", activation="relu")(x)

  # Flatten layer
  x = layers.Flatten()(x)

  # Fully connected layers with dropout
  x = layers.Dense(units=32, activation="relu")(x)
  x = layers.Dropout(rate=0.2)(x)
  outputs = layers.Dense(units=10, activation="softmax")(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model

# Compile and train the model
model = dl_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)