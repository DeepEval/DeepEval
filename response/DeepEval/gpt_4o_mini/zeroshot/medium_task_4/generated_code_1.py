import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the model
    input_layer = layers.Input(shape=(32, 32, 3))

    # Pathway 1: Two convolutional blocks followed by average pooling
    path1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    path1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(path1)
    path1 = layers.AveragePooling2D(pool_size=(2, 2))(path1)

    path1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(path1)
    path1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(path1)
    path1 = layers.AveragePooling2D(pool_size=(2, 2))(path1)

    # Pathway 2: Single convolutional layer
    path2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

    # Combine both pathways
    combined = layers.Add()([path1, path2])

    # Flatten the combined output
    flattened = layers.Flatten()(combined)

    # Fully connected layer to map to 10 classes
    dense_output = layers.Dense(10, activation='softmax')(flattened)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=dense_output)

    return model

# Example of how to use the function to create the model
model = dl_model()
model.summary()  # To view the model architecture