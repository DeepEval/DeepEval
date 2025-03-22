import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Other necessary packages
import numpy as np
import matplotlib.pyplot as plt


def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First branch
    branch1 = layers.GlobalAveragePooling2D()(input_layer)
    branch1 = layers.Dense(64, activation='relu')(branch1)
    branch1 = layers.Dense(64, activation='relu')(branch1)
    branch1 = layers.Reshape(target_shape=(1, 64))(branch1)

    # Second branch
    branch2 = layers.GlobalAveragePooling2D()(input_layer)
    branch2 = layers.Dense(64, activation='relu')(branch2)
    branch2 = layers.Dense(64, activation='relu')(branch2)
    branch2 = layers.Reshape(target_shape=(1, 64))(branch2)

    # Concatenate both branches
    concatenated = layers.Concatenate()([branch1, branch2])

    # Flatten layer
    flattened = layers.Flatten()(concatenated)

    # Fully connected layer
    fully_connected = layers.Dense(10, activation='softmax')(flattened)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=fully_connected)

    return model