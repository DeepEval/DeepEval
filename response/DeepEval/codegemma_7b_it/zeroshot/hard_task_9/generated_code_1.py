import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))
    tower_1x1 = layers.Conv2D(64, (1, 1), padding="same")(inputs)
    tower_1x1 = layers.Activation("relu")(tower_1x1)

    tower_3x3 = layers.Conv2D(64, (1, 1), padding="same")(inputs)
    tower_3x3 = layers.Activation("relu")(tower_3x3)
    tower_3x3 = layers.Conv2D(64, (3, 3), padding="same")(tower_3x3)
    tower_3x3 = layers.Activation("relu")(tower_3x3)

    tower_5x5 = layers.Conv2D(64, (1, 1), padding="same")(inputs)
    tower_5x5 = layers.Activation("relu")(tower_5x5)
    tower_5x5 = layers.Conv2D(64, (3, 3), padding="same")(tower_5x5)
    tower_5x5 = layers.Activation("relu")(tower_5x5)
    tower_5x5 = layers.Conv2D(64, (3, 3), padding="same")(tower_5x5)
    tower_5x5 = layers.Activation("relu")(tower_5x5)

    merged = layers.concatenate([tower_1x1, tower_3x3, tower_5x5])
    merged = layers.Conv2D(32, (1, 1), padding="same")(merged)
    merged = layers.Activation("relu")(merged)

    shortcut = layers.Conv2D(32, (1, 1), padding="same")(inputs)
    shortcut = layers.Activation("relu")(shortcut)

    final_layer = layers.add([merged, shortcut])
    final_layer = layers.Activation("relu")(final_layer)

    x = layers.Flatten()(final_layer)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model