import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial convolution
    conv_initial = layers.Conv2D(32, (3, 3), padding='same')(inputs)

    # Parallel blocks
    block1 = layers.Conv2D(32, (3, 3), padding='same')(conv_initial)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.ReLU()(block1)

    block2 = layers.Conv2D(32, (5, 5), padding='same')(conv_initial)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.ReLU()(block2)

    block3 = layers.Conv2D(32, (7, 7), padding='same')(conv_initial)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.ReLU()(block3)

    # Add outputs of parallel blocks to initial convolution
    added = layers.add([conv_initial, block1, block2, block3])

    # Flatten and fully connected layers
    flattened = layers.Flatten()(added)
    fc1 = layers.Dense(64, activation='relu')(flattened)
    fc2 = layers.Dense(10, activation='softmax')(fc1)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=fc2)

    return model