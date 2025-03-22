import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    for _ in range(3):
        x = layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
    x_branch = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x_branch = layers.BatchNormalization()(x_branch)
    x_branch = keras.activations.relu(x_branch)
    x = layers.Add()([x, x_branch])
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model