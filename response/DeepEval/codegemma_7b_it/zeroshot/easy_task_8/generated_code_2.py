import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
  model = tf.keras.Sequential(
    [
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
            use_bias=False,
        ),
        layers.DepthwiseConv2D(
            (3, 3), padding="same", activation="relu", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.Conv2D(64, (1, 1), padding="same", activation="relu", use_bias=False),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
  )

  return model