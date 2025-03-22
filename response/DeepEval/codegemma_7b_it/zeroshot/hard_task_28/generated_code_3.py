import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Branch path
    branch = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(inputs)
    branch = layers.BatchNormalization()(branch)
    branch = layers.Activation('relu')(branch)

    # Combine paths
    combined = layers.Add()([x, branch])

    # Output layers
    combined = layers.Flatten()(combined)
    outputs = layers.Dense(units=10, activation='softmax')(combined)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model