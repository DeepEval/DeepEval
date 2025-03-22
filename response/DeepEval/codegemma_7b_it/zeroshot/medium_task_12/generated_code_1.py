import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 2
    y = layers.Conv2D(filters=32, kernel_size=3, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    # Concatenate the outputs of Block 1 and Block 2
    concatenated = layers.concatenate([x, y])

    # Block 3
    z = layers.Conv2D(filters=64, kernel_size=3, padding='same')(concatenated)
    z = layers.BatchNormalization()(z)
    z = layers.ReLU()(z)

    # Flatten and pass through fully connected layers
    flattened = layers.Flatten()(z)
    outputs = layers.Dense(10, activation='softmax')(flattened)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model