import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer for the image data
    inputs = keras.Input(shape=(28, 28, 1))

    # Main path
    x = inputs
    for _ in range(3):
        x = layers.Activation('relu')(x)
        x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=32, kernel_size=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, inputs])

    # Branch path
    branch = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
    branch = layers.BatchNormalization()(branch)

    # Fusion layer
    merged = layers.Add()([x, branch])

    # Output layer
    outputs = layers.Flatten()(merged)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Create the model
    model = keras.Model(inputs, outputs)

    return model