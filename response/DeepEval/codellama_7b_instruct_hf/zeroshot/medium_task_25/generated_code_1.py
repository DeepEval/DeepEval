import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Path 1
    path1 = layers.Conv2D(16, (1, 1), activation='relu')(input_layer)

    # Path 2
    path2 = layers.AveragePooling2D((2, 2))(input_layer)
    path2 = layers.Conv2D(16, (1, 1), activation='relu')(path2)

    # Path 3
    path3 = layers.Conv2D(16, (1, 1), activation='relu')(input_layer)
    path3 = layers.Conv2D(16, (3, 3), activation='relu')(path3)
    path3 = layers.Conv2D(16, (3, 3), activation='relu')(path3)

    # Path 4
    path4 = layers.Conv2D(16, (1, 1), activation='relu')(input_layer)
    path4 = layers.Conv2D(16, (3, 3), activation='relu')(path4)
    path4 = layers.Conv2D(16, (3, 3), activation='relu')(path4)

    # Concatenate paths
    x = layers.Concatenate()([path1, path2, path3, path4])

    # Flatten
    x = layers.Flatten()(x)

    # Output layer
    output = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model