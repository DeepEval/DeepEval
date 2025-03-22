import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    y = layers.Conv2D(64, (3, 3), padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.MaxPooling2D()(y)

    # Block 3
    z = layers.Conv2D(128, (3, 3), padding="same")(y)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling2D()(z)

    # Concatenate outputs from all blocks
    concat = layers.concatenate([x, y, z])

    # Flatten and pass through fully connected layers
    flatten = layers.Flatten()(concat)
    dense = layers.Dense(512, activation="relu")(flatten)
    outputs = layers.Dense(10, activation="softmax")(dense)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model