import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(inputs):
    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)  # 3x3 convolutional layer
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.Concatenate()([inputs, x])  # Concatenate along the channel dimension

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 color channel

    # First pathway
    x1 = inputs
    for _ in range(3):  # Repeat block structure three times
        x1 = conv_block(x1)

    # Second pathway
    x2 = inputs
    for _ in range(3):  # Repeat block structure three times
        x2 = conv_block(x2)

    # Merge the two pathways
    merged = layers.Concatenate()([x1, x2])

    # Flatten the merged output
    x = layers.Flatten()(merged)

    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for digits 0-9

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Display the model architecture