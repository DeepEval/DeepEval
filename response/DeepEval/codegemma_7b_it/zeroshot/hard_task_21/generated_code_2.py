import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x1 = [
        layers.Conv2D(32, (1, 1), padding="same", activation="relu")(x)
        for x in x1
    ]

    # Branch path
    x2 = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(inputs)
    x2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x2)

    # Concatenate outputs
    outputs = layers.concatenate(x1 + [x2])

    # Fully connected layers
    outputs = layers.GlobalAveragePooling2D()(outputs)
    outputs = layers.Dense(64, activation="relu")(outputs)
    outputs = layers.Dense(10, activation="softmax")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = dl_model()