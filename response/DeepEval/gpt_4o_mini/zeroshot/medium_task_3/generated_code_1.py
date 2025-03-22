import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    inputs = layers.Input(shape=input_shape)

    # First convolutional block
    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Second convolutional block
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Combine outputs with addition
    combined = layers.add([x1, x2])

    # Fully connected layer for classification
    combined_flat = layers.Flatten()(combined)
    outputs = layers.Dense(10, activation='softmax')(combined_flat)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # To visualize the model structure