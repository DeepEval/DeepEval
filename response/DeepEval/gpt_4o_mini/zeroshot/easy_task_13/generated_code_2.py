import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model(input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape)

    # First 1x1 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)

    # Second 1x1 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # 3x1 convolutional layer with dropout
    x = layers.Conv2D(32, (3, 1), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # 1x3 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Restore channels to match the input's channel count with 1x1 convolution
    x = layers.Conv2D(1, (1, 1), activation='relu')(x)

    # Adding the processed features to the original input
    x = layers.add([x, inputs])

    # Flattening layer
    x = layers.Flatten()(x)

    # Fully connected layer to produce the final probability distribution
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()