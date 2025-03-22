import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    concat1 = layers.Concatenate(axis=-1)([input_layer, conv1])  # Concatenate along channel dimension

    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    concat2 = layers.Concatenate(axis=-1)([concat1, conv2])  # Concatenate along channel dimension

    # Third convolutional layer
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)
    concat3 = layers.Concatenate(axis=-1)([concat2, conv3])  # Concatenate along channel dimension

    # Flattening the output
    flatten = layers.Flatten()(concat3)

    # Fully connected layers
    dense1 = layers.Dense(256, activation='relu')(flatten)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Creating the model
    model = models.Model(inputs=input_layer, outputs=dense2)

    return model

# Example usage:
model = dl_model()
model.summary()