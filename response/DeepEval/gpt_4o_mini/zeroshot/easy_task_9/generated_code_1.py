import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for 28x28 grayscale images (MNIST)
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel

    # 1x1 Convolutional layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)  # Increase channels to 32

    # 3x3 Depthwise Separable Convolutional layer for feature extraction
    x = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)  # Extract features

    # 1x1 Convolutional layer to reduce dimensionality
    x = layers.Conv2D(16, (1, 1), activation='relu')(x)  # Reduce channels to 16

    # Adding the output of this layer to the original input layer (skip connection)
    x = layers.Add()([x, inputs])  # Skip connection

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer to generate final classification probabilities
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST digits

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()  # Display the model architecture