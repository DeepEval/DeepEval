import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    # Block 2
    x2 = layers.Conv2D(128, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    # Block 3
    x3 = layers.Conv2D(256, (3, 3), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    # Parallel branch
    y = layers.Conv2D(256, (3, 3), padding='same')(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    # Concatenate outputs
    concat = layers.concatenate([x1, x2, x3, y])

    # Fully connected layers
    concat = layers.Flatten()(concat)
    outputs = layers.Dense(10, activation='softmax')(concat)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model