import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Create the input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (7, 7), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for filters in [64, 128, 256]:
        x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

    # Branch path
    y = layers.Conv2D(32, (7, 7), padding='same', use_bias=False)(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    # Combine the paths
    combined = layers.Add()([x, y])

    # Classification layers
    combined = layers.Flatten()(combined)
    combined = layers.Dense(512, activation='relu')(combined)
    combined = layers.Dense(10, activation='softmax')(combined)

    # Create the model
    model = keras.Model(inputs, combined)

    return model